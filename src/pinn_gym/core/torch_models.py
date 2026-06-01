"""PyTorch models for POLMI surrogate training."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PolmiSurrogate(nn.Module):
    """Multi-task surrogate: design features -> scalars + force curve."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 384,
        blocks: int = 6,
        dropout: float = 0.04,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.blocks(z)
        return self.head(z)
