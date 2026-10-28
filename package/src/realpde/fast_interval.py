from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FastIntervalConfig:
    diagnostic_channels: int
    hidden_channels: int = 48
    out_steps: int = 20
    maximum_normalized_shift: float = 1.0
    minimum_normalized_radius: float = 1e-4
    maximum_normalized_radius: float = 4.0
    initial_log_radius: float = -1.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FastIntervalConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FastIntervalHead(nn.Module):
    """Cheap center/radius student operating on ordinal diagnostics."""

    def __init__(self, config: FastIntervalConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Conv2d(config.diagnostic_channels, config.hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(
                config.hidden_channels,
                config.hidden_channels,
                3,
                padding=1,
                groups=config.hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.out_steps * 4, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            bias = self.network[-1].bias.view(config.out_steps, 2, 2)
            bias[:, 0].zero_()
            bias[:, 1].fill_(float(config.initial_log_radius))

    def forward(self, diagnostics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if diagnostics.ndim != 4:
            raise ValueError("diagnostics must have shape (N,C,H,W)")
        if diagnostics.shape[1] != self.config.diagnostic_channels:
            raise ValueError(
                f"expected {self.config.diagnostic_channels} diagnostic channels, "
                f"got {diagnostics.shape[1]}"
            )
        raw = self.network(diagnostics)
        batch, _, height, width = raw.shape
        decisions = raw.reshape(
            batch, self.config.out_steps, 2, 2, height, width
        )
        shift = decisions[:, :, 0].permute(0, 1, 3, 4, 2)
        shift = torch.tanh(shift) * float(self.config.maximum_normalized_shift)
        log_radius = decisions[:, :, 1].permute(0, 1, 3, 4, 2)
        radius = log_radius.clamp(
            min=float(torch.log(torch.tensor(self.config.minimum_normalized_radius))),
            max=float(torch.log(torch.tensor(self.config.maximum_normalized_radius))),
        ).exp()
        return shift, radius
