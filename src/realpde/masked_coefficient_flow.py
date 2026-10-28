from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock
from .model import ModelConfig, MultiScaleUNetContext


@dataclass(frozen=True)
class MaskedCoefficientFlowConfig:
    diagnostic_channels: int = 186
    coefficient_channels: int = 24
    width: int = 64
    condition_dilations: tuple[int, ...] = (1, 2, 4, 8)
    denoiser_dilations: tuple[int, ...] = (1, 2, 4, 8)
    time_channels: int = 16

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MaskedConditionalCoefficientFlow(nn.Module):
    """Conditional rectified flow over masked temporal coefficient maps."""

    def __init__(self, config: MaskedCoefficientFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedCoefficientFlowConfig()
        self.condition_lift = nn.Conv2d(
            self.config.diagnostic_channels + 1, self.config.width, 1
        )
        self.condition_blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation)
            for dilation in self.config.condition_dilations
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
        self.condition_context = MultiScaleUNetContext(context_config)
        self.velocity_lift = nn.Conv2d(
            self.config.width + self.config.coefficient_channels + self.config.time_channels,
            self.config.width,
            1,
        )
        self.velocity_blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation)
            for dilation in self.config.denoiser_dilations
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.velocity_head = nn.Sequential(
            nn.GroupNorm(groups, self.config.width),
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, self.config.coefficient_channels, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def condition(self, diagnostics: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.condition_lift(
            torch.cat((diagnostics.float(), mask[:, None].float()), dim=1)
        )
        for block in self.condition_blocks:
            hidden = block(hidden)
        return self.condition_context(hidden)

    def time_embedding(self, time_value: torch.Tensor) -> torch.Tensor:
        half = self.config.time_channels // 2
        frequency = 2.0 ** torch.arange(
            half, device=time_value.device, dtype=time_value.dtype
        )
        phase = torch.pi * time_value[:, None] * frequency[None]
        embedded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        return embedded[:, :, None, None].expand(-1, -1, 32, 64)

    def velocity(
        self,
        latent: torch.Tensor,
        time_value: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.velocity_lift(
            torch.cat((condition, latent, self.time_embedding(time_value)), dim=1)
        )
        for block in self.velocity_blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden) * mask[:, None]

