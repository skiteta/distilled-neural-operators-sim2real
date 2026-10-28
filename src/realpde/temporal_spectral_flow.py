from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalSpectralFlowConfig:
    out_steps: int = 20
    frequency_bins: int = 6
    spatial_pool: int = 4
    width: int = 96
    depth: int = 6
    ensemble_size: int = 4
    integration_steps: int = 4
    noise_seed: int = 20260811

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TemporalSpectralFlowConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(source)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return source + hidden


class TemporalSpectralResidualFlow(nn.Module):
    """Rectified flow over low-spatial, temporal-Fourier forecast residuals."""

    def __init__(self, config: TemporalSpectralFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalSpectralFlowConfig()
        self.latent_channels = self.config.frequency_bins * 2 * 2
        self.latent_height = 32 // self.config.spatial_pool
        self.latent_width = 64 // self.config.spatial_pool
        condition_inputs = 2 * self.latent_channels + 2 + 4
        self.condition_lift = nn.Conv2d(condition_inputs, self.config.width, 1)
        self.condition_blocks = nn.ModuleList(
            ResidualBlock(self.config.width) for _ in range(3)
        )
        self.velocity_lift = nn.Conv2d(
            self.latent_channels + self.config.width + 3,
            self.config.width,
            1,
        )
        self.velocity_blocks = nn.ModuleList(
            ResidualBlock(self.config.width) for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.velocity_norm = nn.GroupNorm(groups, self.config.width)
        self.velocity_head = nn.Conv2d(self.config.width, self.latent_channels, 1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

        self.register_buffer(
            "latent_mean",
            torch.zeros(1, self.latent_channels, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "latent_scale",
            torch.ones(1, self.latent_channels, 1, 1),
            persistent=True,
        )
        yy = torch.linspace(-1.0, 1.0, self.latent_height)
        xx = torch.linspace(-1.0, 1.0, self.latent_width)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        self.register_buffer(
            "coordinates",
            torch.stack((grid_x, grid_y), dim=0),
            persistent=True,
        )
        generator = torch.Generator().manual_seed(self.config.noise_seed)
        noise = torch.randn(
            self.config.ensemble_size,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            generator=generator,
        )
        for index in range(1, self.config.ensemble_size, 2):
            noise[index].copy_(-noise[index - 1])
        self.register_buffer("base_noise", noise, persistent=True)

    @staticmethod
    def statistics(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = observed[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return mean, std

    def pack(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 5 or field.shape[1] != self.config.out_steps or field.shape[-1] != 2:
            raise ValueError(f"unexpected temporal field {tuple(field.shape)}")
        batch, steps, height, width, channels = field.shape
        pooled = F.avg_pool2d(
            field.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width),
            self.config.spatial_pool,
        )
        pooled = pooled.reshape(
            batch, steps, channels, self.latent_height, self.latent_width
        )
        spectrum = torch.fft.rfft(pooled.float(), dim=1, norm="ortho")
        selected = spectrum[:, : self.config.frequency_bins]
        packed = torch.view_as_real(selected).permute(0, 1, 2, 5, 3, 4).contiguous()
        return packed.reshape(
            batch, self.latent_channels, self.latent_height, self.latent_width
        )

    def unpack(self, latent: torch.Tensor, height: int = 32, width: int = 64) -> torch.Tensor:
        if latent.shape[1:] != (
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        ):
            raise ValueError(f"unexpected temporal latent {tuple(latent.shape)}")
        batch = latent.shape[0]
        pairs = latent.reshape(
            batch,
            self.config.frequency_bins,
            2,
            2,
            self.latent_height,
            self.latent_width,
        )
        pairs = pairs.permute(0, 1, 2, 4, 5, 3).contiguous()
        selected = torch.view_as_complex(pairs)
        spectrum = torch.zeros(
            batch,
            self.config.out_steps // 2 + 1,
            2,
            self.latent_height,
            self.latent_width,
            device=latent.device,
            dtype=selected.dtype,
        )
        spectrum[:, : self.config.frequency_bins] = selected
        pooled = torch.fft.irfft(
            spectrum, n=self.config.out_steps, dim=1, norm="ortho"
        )
        upsampled = F.interpolate(
            pooled.reshape(batch * self.config.out_steps, 2, self.latent_height, self.latent_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.reshape(batch, self.config.out_steps, 2, height, width).permute(
            0, 1, 3, 4, 2
        )

    def standardize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_scale.clamp_min(1e-5)

    def destandardize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_scale + self.latent_mean

    def set_latent_statistics(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != self.latent_mean.shape or scale.shape != self.latent_scale.shape:
            raise ValueError("latent statistics have the wrong shape")
        self.latent_mean.copy_(mean)
        self.latent_scale.copy_(scale.clamp_min(1e-4))

    def condition(
        self, observed: torch.Tensor, point: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self.statistics(observed)
        observed_normalized = (observed[..., :2].float() - mean) / std
        point_residual = (point[..., :2].float() - baseline[..., :2].float()) / std
        observed_spectrum = self.pack(observed_normalized)
        point_spectrum = self.pack(point_residual)
        final_mean = observed_normalized[:, -1].mean(dim=(1, 2))
        trend = (observed_normalized[:, -1] - observed_normalized[:, -5]).mean(
            dim=(1, 2)
        ) / 4.0
        descriptors = torch.cat((final_mean, trend), dim=1)[:, :, None, None].expand(
            -1, -1, self.latent_height, self.latent_width
        )
        coordinates = self.coordinates[None].expand(observed.shape[0], -1, -1, -1)
        hidden = self.condition_lift(
            torch.cat(
                (observed_spectrum, point_spectrum, coordinates, descriptors), dim=1
            )
        )
        for block in self.condition_blocks:
            hidden = block(hidden)
        return hidden

    @staticmethod
    def time_features(time_value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        phase = 2.0 * torch.pi * time_value
        features = torch.stack((time_value, torch.sin(phase), torch.cos(phase)), dim=1)
        return features[:, :, None, None].expand(-1, -1, height, width)

    def velocity(
        self, latent: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        features = self.time_features(time_value, latent.shape[-2], latent.shape[-1]).to(
            latent.dtype
        )
        hidden = self.velocity_lift(torch.cat((latent, condition, features), dim=1))
        for block in self.velocity_blocks:
            hidden = block(hidden)
        return self.velocity_head(F.silu(self.velocity_norm(hidden)))

    def flow_matching(
        self,
        observed: torch.Tensor,
        point: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, std = self.statistics(observed)
        target_residual = (target[..., :2].float() - point[..., :2].float()) / std
        target_latent = self.standardize(self.pack(target_residual))
        noise = torch.randn_like(target_latent)
        time_value = torch.rand(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        shaped_time = time_value[:, None, None, None]
        interpolated = (1.0 - shaped_time) * noise + shaped_time * target_latent
        condition = self.condition(observed, point, baseline)
        predicted_velocity = self.velocity(interpolated, time_value, condition)
        target_velocity = target_latent - noise
        endpoint = interpolated + (1.0 - shaped_time) * predicted_velocity
        endpoint_residual = self.unpack(self.destandardize(endpoint)) * std
        candidate_uv = point[..., :2].float() + endpoint_residual
        candidate = torch.cat((candidate_uv, torch.zeros_like(candidate_uv[..., :1])), dim=-1)
        return candidate, {
            "flow_matching": (predicted_velocity - target_velocity).square().mean(),
            "latent_endpoint": (endpoint - target_latent).square().mean(),
        }

    def integrate(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        result = latent
        steps = self.config.integration_steps
        for index in range(steps):
            time_value = result.new_full((result.shape[0],), index / steps)
            result = result + self.velocity(result, time_value, condition) / steps
        return result

    def sample_mean(
        self, observed: torch.Tensor, point: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        condition = self.condition(observed, point, baseline)
        batch = observed.shape[0]
        ensemble = self.config.ensemble_size
        initial = self.base_noise[None].expand(batch, -1, -1, -1, -1).reshape(
            batch * ensemble,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        repeated_condition = condition[:, None].expand(-1, ensemble, -1, -1, -1).reshape(
            batch * ensemble,
            condition.shape[1],
            self.latent_height,
            self.latent_width,
        )
        endpoint = self.integrate(initial, repeated_condition)
        residual = self.unpack(self.destandardize(endpoint))
        residual = residual.reshape(batch, ensemble, self.config.out_steps, 32, 64, 2)
        _, std = self.statistics(observed)
        corrected_uv = point[..., :2] + residual.mean(dim=1) * std
        return torch.cat((corrected_uv, torch.zeros_like(corrected_uv[..., :1])), dim=-1)


def project_local_tke(candidate: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    candidate_uv = candidate[..., :2].float()
    baseline_uv = baseline[..., :2].float()
    candidate_mean = candidate_uv.mean(dim=1, keepdim=True)
    candidate_fluctuation = candidate_uv - candidate_mean
    baseline_fluctuation = baseline_uv - baseline_uv.mean(dim=1, keepdim=True)
    candidate_energy = candidate_fluctuation.square().sum(dim=-1).mean(dim=1, keepdim=True)
    baseline_energy = baseline_fluctuation.square().sum(dim=-1).mean(dim=1, keepdim=True)
    factor = (baseline_energy / candidate_energy.clamp_min(1e-12)).clamp_min(0.0).sqrt()
    projected_uv = candidate_mean + candidate_fluctuation * factor[..., None]
    return torch.cat((projected_uv, torch.zeros_like(projected_uv[..., :1])), dim=-1)


class TemporalSpectralFlowEndpoint(nn.Module):
    def __init__(self, point_refiner: nn.Module, flow: TemporalSpectralResidualFlow) -> None:
        super().__init__()
        self.point_refiner = point_refiner
        self.flow = flow

    def point(self, observed: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        return project_local_tke(self.point_refiner(observed, baseline), baseline)

    def forward(self, observed: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        point = self.point(observed, baseline)
        candidate = self.flow.sample_mean(observed, point, baseline)
        return project_local_tke(candidate, baseline)
