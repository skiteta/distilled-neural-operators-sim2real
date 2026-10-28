from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock


@dataclass(frozen=True)
class TernaryIntervalActionConfig:
    diagnostic_channels: int = 186
    state_channels: int = 80
    width: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TernaryIntervalActionClassifier(nn.Module):
    """Predict outside and conditional sign logits for 20x2 interval fields."""

    def __init__(self, config: TernaryIntervalActionConfig | None = None) -> None:
        super().__init__()
        self.config = config or TernaryIntervalActionConfig()
        self.lift = nn.Conv2d(
            self.config.diagnostic_channels + self.config.state_channels,
            self.config.width,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation)
            for dilation in self.config.dilations
        )
        self.head = nn.Conv2d(self.config.width, 80, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        diagnostics: torch.Tensor,
        current_center: torch.Tensor,
        normalized_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (20, 32, 64, 2)
        if diagnostics.ndim != 4:
            raise ValueError("diagnostics must have shape (N,C,H,W)")
        if current_center.shape[1:] != expected or normalized_radius.shape[1:] != expected:
            raise ValueError("center and radius must have shape (N,20,32,64,2)")
        center_channels = current_center.permute(0, 1, 4, 2, 3).reshape(
            len(current_center), 40, 32, 64
        )
        radius_channels = normalized_radius.permute(0, 1, 4, 2, 3).reshape(
            len(normalized_radius), 40, 32, 64
        )
        hidden = self.lift(
            torch.cat((diagnostics.float(), center_channels, radius_channels), dim=1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(hidden).reshape(len(hidden), 2, 20, 2, 32, 64)
        outside = raw[:, 0].permute(0, 1, 3, 4, 2).contiguous()
        sign = raw[:, 1].permute(0, 1, 3, 4, 2).contiguous()
        return outside, sign

