from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock
from .model import ModelConfig, MultiScaleUNetContext


@dataclass(frozen=True)
class MaskedResidualSpecialistConfig:
    diagnostic_channels: int = 186
    width: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    out_steps: int = 20
    velocity_channels: int = 2
    maximum_normalized_residual: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MaskedTrajectoryResidualSpecialist(nn.Module):
    """Predict a local trajectory residual and hard-mask it at deployment."""

    def __init__(self, config: MaskedResidualSpecialistConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedResidualSpecialistConfig()
        self.lift = nn.Conv2d(self.config.diagnostic_channels + 1, self.config.width, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation)
            for dilation in self.config.dilations
        )
        context_config = ModelConfig(
            width=self.config.width,
            depth=2,
            spectral_rank=12,
            modes_y=10,
            modes_x=18,
            expansion=2,
            dropout=0.0,
            use_pressure=False,
        )
        self.multiscale = MultiScaleUNetContext(context_config)
        groups = 8 if self.config.width % 8 == 0 else 1
        output_channels = self.config.out_steps * self.config.velocity_channels
        self.head = nn.Sequential(
            nn.GroupNorm(groups, self.config.width),
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, output_channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, diagnostics: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if diagnostics.ndim != 4 or diagnostics.shape[1] != self.config.diagnostic_channels:
            raise ValueError("unexpected diagnostics shape")
        if mask.shape != (len(diagnostics), 32, 64):
            raise ValueError("mask must have shape (N,32,64)")
        hidden = self.lift(torch.cat((diagnostics.float(), mask[:, None].float()), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.multiscale(hidden)
        raw = self.head(hidden)
        batch, _, height, width = raw.shape
        residual = raw.reshape(
            batch,
            self.config.out_steps,
            self.config.velocity_channels,
            height,
            width,
        ).permute(0, 1, 3, 4, 2)
        maximum = float(self.config.maximum_normalized_residual)
        residual = maximum * torch.tanh(residual / maximum)
        return residual * mask[:, None, :, :, None]

    @staticmethod
    def apply(
        point: torch.Tensor,
        normalized_residual: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return point.float() + normalized_residual.float() * scale.float()

