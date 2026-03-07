"""
CNN Model - Version 2.

Improvements over v1:
- Multi-scale parallel convolutions (kernels 3, 7, 15) capture short & long-range
  patterns in the B waveform simultaneously.
- Residual convolutional blocks with Squeeze-and-Excite channel attention.
- FiLM (Feature-wise Linear Modulation): Freq/Temp/Hdc scalars directly modulate
  CNN feature maps at two depths — not just appended at the end.
- Global Average + Global Max Pooling concatenated for richer sequence summary.
- Log10 Loss target at config/training level (model outputs raw scalar).
- Kaiming weight initialization.
"""

import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Channel attention — re-weights feature maps based on global context."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Projects scalars → (gamma, beta) to modulate CNN feature maps:
        output = (1 + gamma) * x + beta
    """

    def __init__(self, scalar_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(scalar_dim, num_channels * 2),
            nn.GELU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x, scalars):
        # x: (B, C, L),  scalars: (B, scalar_dim)
        gamma, beta = self.proj(scalars).chunk(2, dim=-1)   # each (B, C)
        return (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)


class ResConvBlock(nn.Module):
    """1-D Residual Conv Block with Squeeze-and-Excite."""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.se  = SqueezeExcite(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.block(x) + x))


class MultiScaleConv(nn.Module):
    """Parallel convolutions with different kernel sizes, concatenated."""

    def __init__(self, in_channels: int, out_channels: int, kernels=(3, 7, 15)):
        super().__init__()
        branch_ch = out_channels // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_ch, k, padding=k // 2),
                nn.BatchNorm1d(branch_ch),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class CNNNetwork(nn.Module):
    """
    Multi-scale CNN for power loss prediction from B waveforms (v2).

    Architecture:
        B waveform
          → MultiScaleConv (k=3,7,15)
          → FiLM-1 (early scalar injection)
          → ResConvBlock × num_layers
          → FiLM-2 (late scalar injection)
          → AvgPool + MaxPool (concat)
          → MLP head (+ raw scalars)
          → output (1)

    Args:
        input_dim   (int):  Waveform input channels. Default 1.
        num_channels(int):  CNN channel width (must be divisible by 3). Default 96.
        num_layers  (int):  Number of ResConvBlocks. Default 4.
        scalar_dim  (int):  Number of scalar features. Default 3.
        dropout     (float):Dropout rate. Default 0.15.
        stats       (dict): Legacy compat — normalization stats, not used internally.
    """

    def __init__(self, input_dim=1, num_channels=96, num_layers=4,
                 scalar_dim=3, dropout=0.15, stats=None):
        super().__init__()
        self.stats = stats or {}

        self.ms_conv  = MultiScaleConv(input_dim, num_channels, kernels=(3, 7, 15))
        self.film1    = FiLM(scalar_dim, num_channels)
        self.res_blocks = nn.ModuleList([
            ResConvBlock(num_channels, kernel_size=7, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.film2 = FiLM(scalar_dim, num_channels)

        # Pooled dim = avg + max = 2 × num_channels; plus raw scalars appended
        head_in = num_channels * 2 + scalar_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, b_seq: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            b_seq   : (batch, seq_len, 1)
            scalars : (batch, 3)  — [Frequency, Temperature, Hdc]
        Returns:
            out     : (batch, 1)  — predicted log10(Power Loss)
        """
        x = b_seq.permute(0, 2, 1)             # (B, 1, L)
        x = self.ms_conv(x)                     # (B, C, L)
        x = self.film1(x, scalars)              # FiLM modulation
        for block in self.res_blocks:
            x = block(x)
        x = self.film2(x, scalars)              # second FiLM modulation
        pooled = torch.cat([x.mean(-1), x.max(-1).values], dim=-1)  # (B, 2C)
        return self.head(torch.cat([pooled, scalars], dim=-1))       # (B, 1)


if __name__ == "__main__":
    model = CNNNetwork()
    B = torch.randn(16, 1024, 1)
    s = torch.randn(16, 3)
    print(f"Output: {model(B, s).shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")