from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock


@dataclass(frozen=True)
class CategoricalEnergyConfig:
    diagnostic_channels: int = 186
    energy_channels: int = 1
    width: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CategoricalEnergyClassifier(nn.Module):
    """Predict decrease, keep, or increase for each local future-TKE pixel."""

    def __init__(self, config: CategoricalEnergyConfig | None = None) -> None:
        super().__init__()
        self.config = config or CategoricalEnergyConfig()
        self.lift = nn.Conv2d(
            self.config.diagnostic_channels + self.config.energy_channels,
            self.config.width,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation)
            for dilation in self.config.dilations
        )
        self.head = nn.Conv2d(self.config.width, 3, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = 5.0

    def forward(self, diagnostics: torch.Tensor, normalized_log_tke: torch.Tensor) -> torch.Tensor:
        if diagnostics.ndim != 4:
            raise ValueError("diagnostics must have shape (N,C,H,W)")
        if normalized_log_tke.shape[1:] != (1, 32, 64):
            raise ValueError("normalized_log_tke must have shape (N,1,32,64)")
        hidden = self.lift(torch.cat((diagnostics.float(), normalized_log_tke.float()), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)

