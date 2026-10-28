from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .frequency_tke_head import DilatedResidualBlock
from .model import ModelConfig, MultiScaleUNetContext


@dataclass(frozen=True)
class MaskedSpectralResidualConfig:
    diagnostic_channels: int = 186
    width: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    bins: int = 6
    velocity_channels: int = 2
    out_steps: int = 20

    @property
    def coefficient_channels(self) -> int:
        return self.bins * self.velocity_channels * 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MaskedSpectralResidualSpecialist(nn.Module):
    """Predict low-temporal-frequency complex residual coefficients."""

    def __init__(self, config: MaskedSpectralResidualConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedSpectralResidualConfig()
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
        self.head = nn.Sequential(
            nn.GroupNorm(groups, self.config.width),
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, self.config.coefficient_channels, 1),
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
        return self.head(hidden) * mask[:, None]

    def pack(self, residual: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(residual.float(), dim=1, norm="ortho")[:, : self.config.bins]
        packed = torch.view_as_real(spectrum).permute(0, 1, 4, 5, 2, 3)
        return packed.reshape(len(residual), self.config.coefficient_channels, 32, 64)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        batch = len(packed)
        shaped = packed.reshape(
            batch,
            self.config.bins,
            self.config.velocity_channels,
            2,
            32,
            64,
        ).permute(0, 1, 4, 5, 2, 3).contiguous()
        selected = torch.view_as_complex(shaped)
        spectrum = torch.zeros(
            batch,
            self.config.out_steps // 2 + 1,
            32,
            64,
            self.config.velocity_channels,
            dtype=selected.dtype,
            device=selected.device,
        )
        spectrum[:, : self.config.bins] = selected
        return torch.fft.irfft(spectrum, n=self.config.out_steps, dim=1, norm="ortho")

    def reconstruct(
        self,
        standardized_coefficients: torch.Tensor,
        coefficient_scale: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        packed = standardized_coefficients * coefficient_scale
        return self.unpack(packed) * mask[:, None, :, :, None]

    @staticmethod
    def apply(point: torch.Tensor, normalized_residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return point.float() + normalized_residual.float() * scale.float()

