from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock


@dataclass(frozen=True)
class MultiscaleCenterConfig:
    diagnostic_channels: int = 186
    center_channels: int = 40
    width: int = 48
    dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    maximum_correction: float = 0.25

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MultiscaleCenterRefiner(nn.Module):
    """Refine a frozen normalized interval center with 43x43 spatial context."""

    def __init__(self, config: MultiscaleCenterConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiscaleCenterConfig()
        input_channels = self.config.diagnostic_channels + self.config.center_channels
        self.lift = nn.Conv2d(input_channels, self.config.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation) for dilation in self.config.dilations
        )
        self.head = nn.Conv2d(self.config.width, self.config.center_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, diagnostics: torch.Tensor, current_center: torch.Tensor) -> torch.Tensor:
        if diagnostics.ndim != 4:
            raise ValueError("diagnostics must have shape (N,C,H,W)")
        if current_center.ndim != 5 or current_center.shape[1:] != (
            20,
            32,
            64,
            2,
        ):
            raise ValueError("current_center must have shape (N,20,32,64,2)")
        center_channels = current_center.permute(0, 1, 4, 2, 3).reshape(
            len(current_center), self.config.center_channels, 32, 64
        )
        hidden = self.lift(torch.cat((diagnostics.float(), center_channels), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(hidden)
        correction = float(self.config.maximum_correction) * torch.tanh(raw)
        return correction.reshape(len(raw), 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
