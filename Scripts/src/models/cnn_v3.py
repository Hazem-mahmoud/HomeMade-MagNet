import torch
import torch.nn as nn


# ============================================================
# Squeeze-and-Excite
# ============================================================
class SqueezeExcite(nn.Module):
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


# ============================================================
# FiLM Layer
# ============================================================
class FiLM(nn.Module):
    def __init__(self, scalar_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(scalar_dim, num_channels * 2),
            nn.GELU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x, scalars):
        gamma, beta = self.proj(scalars).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)


# ============================================================
# Residual Block
# ============================================================
class ResConvBlock(nn.Module):
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

        self.se = SqueezeExcite(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.block(x) + x))


# ============================================================
# Multi-Scale Convolution
# ============================================================
class MultiScaleConv(nn.Module):
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


# ============================================================
# Main Model
# ============================================================
class CNNNetwork(nn.Module):
    """
    Now supports:
    - B waveform
    - H waveform
    - Interaction channel (B * H)
    """

    def __init__(self, input_dim=3, num_channels=96, num_layers=4,
                 scalar_dim=2, dropout=0.15, stats=None):
        super().__init__()

        self.stats = stats or {}

        kernels = (3, 7, 15)
        branch_ch = num_channels // len(kernels)
        actual_ch = branch_ch * len(kernels)

        # Multi-channel input (B, H, B*H)
        self.ms_conv = MultiScaleConv(input_dim, num_channels, kernels=kernels)

        self.film1 = FiLM(scalar_dim, actual_ch)

        self.res_blocks = nn.ModuleList([
            ResConvBlock(actual_ch, kernel_size=7, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.film2 = FiLM(scalar_dim, actual_ch)

        head_in = actual_ch * 2 + scalar_dim

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

    # ========================================================
    # Weight Initialization
    # ========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ========================================================
    # Forward Pass
    # ========================================================
    def forward(self, B: torch.Tensor, H: torch.Tensor, scalars: torch.Tensor):
        """
        Args:
            B        : (batch, seq_len, 1)
            H        : (batch, seq_len, 1)
            scalars  : (batch, 2)  → [freq, temp]
        """

        # ---------------------------
        # Normalize each channel
        # ---------------------------
        B = B / (B.abs().max(dim=1, keepdim=True)[0] + 1e-6)
        H = H / (H.abs().max(dim=1, keepdim=True)[0] + 1e-6)

        # ---------------------------
        # Interaction feature
        # ---------------------------
        BH = B * H

        # ---------------------------
        # Stack channels
        # ---------------------------
        x = torch.cat([B, H, BH], dim=-1)  # (B, L, 3)

        # Conv1D format
        x = x.permute(0, 2, 1)  # (B, 3, L)

        # ---------------------------
        # CNN pipeline
        # ---------------------------
        x = self.ms_conv(x)
        x = self.film1(x, scalars)

        for block in self.res_blocks:
            x = block(x)

        x = self.film2(x, scalars)

        # ---------------------------
        # Pooling
        # ---------------------------
        pooled = torch.cat([x.mean(-1), x.max(-1).values], dim=-1)

        # ---------------------------
        # Head
        # ---------------------------
        return self.head(torch.cat([pooled, scalars], dim=-1))


# ============================================================
# Test Run
# ============================================================
if __name__ == "__main__":
    model = CNNNetwork()

    B = torch.randn(16, 1024, 1)
    H = torch.randn(16, 1024, 1)
    scalars = torch.randn(16, 2)

    out = model(B, H, scalars)

    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")