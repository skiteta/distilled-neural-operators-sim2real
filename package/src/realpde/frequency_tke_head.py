from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FrequencyTKEHeadConfig:
    """Small spatial head for a future/base local-TKE log ratio."""

    width: int = 48
    dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    maximum_log_ratio: float = 1.5
    frequency_bins: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class FrequencyConditionedTKEHead(nn.Module):
    """Predict a local future-TKE ratio from observed spectra and a frozen point.

    The output changes only the magnitude of each pixel's temporal fluctuation.
    It cannot change the point trajectory's temporal mean or local normalized
    fluctuation direction.
    """

    energy_channels = 8
    mean_channels = 4
    availability_channels = 1
    coordinate_channels = 2

    def __init__(self, config: FrequencyTKEHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or FrequencyTKEHeadConfig()
        frequency_channels = 4 * self.config.frequency_bins
        self.feature_channels = (
            frequency_channels
            + self.energy_channels
            + self.mean_channels
            + self.availability_channels
            + self.coordinate_channels
        )
        self.lift = nn.Conv2d(self.feature_channels, self.config.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(self.config.width, dilation) for dilation in self.config.dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        y = torch.linspace(-1.0, 1.0, 32)
        x = torch.linspace(-1.0, 1.0, 64)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("coordinates", torch.stack((grid_y, grid_x), dim=0), persistent=True)

    @staticmethod
    def temporal_tke(flow: torch.Tensor) -> torch.Tensor:
        uv = flow[..., :2].float()
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1)

    def temporal_frequency_features(self, observed: torch.Tensor) -> torch.Tensor:
        uv = observed[..., :2].float()
        centered = uv - uv.mean(dim=1, keepdim=True)
        component_scale = centered.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        normalized = centered / component_scale.clamp_min(1e-5)
        spectrum = torch.fft.rfft(normalized, dim=1, norm="ortho")
        selected = spectrum[:, 1 : self.config.frequency_bins + 1]
        if selected.shape[1] != self.config.frequency_bins:
            raise RuntimeError(f"unexpected observed spectrum {tuple(selected.shape)}")
        packed = torch.view_as_real(selected).permute(0, 1, 4, 5, 2, 3)
        return packed.reshape(observed.shape[0], -1, 32, 64)

    def features(self, observed: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        uv = observed[..., :2].float()
        point_uv = point[..., :2].float()
        component_scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        frequency = self.temporal_frequency_features(observed)
        energies = torch.stack(
            (
                self.temporal_tke(uv),
                self.temporal_tke(uv[:, :10]),
                self.temporal_tke(uv[:, 10:]),
                self.temporal_tke(uv[:, 0:5]),
                self.temporal_tke(uv[:, 5:10]),
                self.temporal_tke(uv[:, 10:15]),
                self.temporal_tke(uv[:, 15:20]),
                self.temporal_tke(point_uv),
            ),
            dim=1,
        )
        log_energies = energies.clamp_min(1e-8).log().clamp(-12.0, 6.0)
        scale = component_scale[:, 0, 0, 0, :, None, None]
        means = torch.cat((uv.mean(dim=1), point_uv.mean(dim=1)), dim=-1)
        means = means.permute(0, 3, 1, 2) / torch.cat((scale, scale), dim=1)
        available = uv.abs().amax(dim=-1).gt(0.0).float().mean(dim=1, keepdim=True)
        coordinates = self.coordinates[None].expand(observed.shape[0], -1, -1, -1)
        result = torch.cat((frequency, log_energies, means, available, coordinates), dim=1)
        if result.shape[1] != self.feature_channels:
            raise RuntimeError(f"unexpected feature shape {tuple(result.shape)}")
        return result

    def forward(self, observed: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(self.features(observed, point))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(hidden)
        maximum = float(self.config.maximum_log_ratio)
        return maximum * torch.tanh(raw / maximum)

    @staticmethod
    def apply_log_ratio(point: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        uv = point[..., :2].float()
        mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - mean
        scale = torch.exp(0.5 * log_ratio).permute(0, 2, 3, 1)
        return mean + fluctuation * scale[:, None]
