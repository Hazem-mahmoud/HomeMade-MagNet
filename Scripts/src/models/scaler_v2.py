"""
Scaler Model - Version 2.

Improvements over v1:
- Deeper network: 5 layers at hidden_dim=256 (was 3 layers at 64)
- BatchNorm1d after each linear for training stability & faster convergence
- Dropout (0.2) to reduce overfitting
- Residual (skip) connections every 2 layers for better gradient flow
- GELU activations (smoother than ReLU, better for regression)
- Kaiming weight initialization
- Log10 Loss normalization (config-driven, see config.yaml)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-layer residual block with BatchNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))   # residual skip


class ScalerNetwork(nn.Module):
    """
    FNN: scalar features → scalar power loss prediction.

    Architecture (v3):
        InputProj → ResidualBlock × N → OutputHead

    Args:
        input_dim  (int):   Number of input features. Default 4 (B_pk, Freq, Temp, Hdc).
        hidden_dim (int):   Hidden layer width. Default 256.
        num_layers (int):   Depth; controls number of residual blocks. Default 5.
        output_dim (int):   Output size. Default 1.
        dropout    (float): Dropout rate. Default 0.2.
    """

    def __init__(self, input_dim=4, hidden_dim=256, num_layers=5,
                 output_dim=1, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        num_res_blocks = max(1, (num_layers - 1) // 2)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:   x : (batch, input_dim)
        Returns:    (batch, output_dim)
        """
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)


if __name__ == "__main__":
    model = ScalerNetwork(input_dim=4, hidden_dim=256, num_layers=5, output_dim=1)
    x = torch.randn(32, 4)
    print(f"Output shape: {model(x).shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")