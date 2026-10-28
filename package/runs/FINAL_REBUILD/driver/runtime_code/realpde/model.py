from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_steps: int = 20
    out_steps: int = 20
    width: int = 128
    depth: int = 10
    spectral_rank: int = 32
    modes_y: int = 12
    modes_x: int = 20
    expansion: int = 3
    dropout: float = 0.03
    # Keep the historical defaults so organizer/checkpoint compatibility is
    # preserved. New sim-to-real models deliberately ignore pressure because
    # real PIV inputs contain no pressure channel.
    use_pressure: bool = True
    use_physics_features: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any], data: dict[str, Any] | None = None) -> ModelConfig:
        values = dict(values)
        if data:
            values.setdefault("in_steps", data.get("in_steps", 20))
            values.setdefault("out_steps", data.get("out_steps", 20))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatentFlowConfig:
    """Configuration for the conditional Fourier-residual probability flow."""

    teacher: dict[str, Any]
    width: int = 96
    depth: int = 6
    modes_y: int = 12
    modes_x: int = 20
    expansion: int = 2
    dropout: float = 0.02
    data_scale: float = 2.0
    sample_steps: int = 3
    ensemble_size: int = 4
    max_ensemble_size: int = 8
    sample_seed: int = 20260809
    point_mode: str = "anchor"
    interval_scale: float = 1.0
    teacher_width_floor: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LatentFlowConfig:
        values = dict(values)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistilledIntervalConfig:
    """One-pass student for aleatoric and cross-regime interval uncertainty."""

    anchor: dict[str, Any]
    width: int = 48
    depth: int = 4
    expansion: int = 2
    dropout: float = 0.02
    interval_scale: float = 1.0
    phase_channels: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DistilledIntervalConfig:
        values = dict(values)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralPosteriorConfig:
    """Configuration for masked joint-trajectory posterior refinement."""

    anchor: dict[str, Any]
    interval_teacher: dict[str, Any] | None = None
    width: int = 32
    depth: int = 5
    expansion: int = 2
    dropout: float = 0.02
    modes_t: int = 10
    modes_y: int = 10
    modes_x: int = 18
    sample_steps: int = 4
    ensemble_size: int = 2
    max_ensemble_size: int = 4
    sample_seed: int = 20260810
    point_correction_scale: float = 0.25
    correction_source: str = "mean"
    interval_scale: float = 1.0
    anchor_width_floor: float = 0.0
    observation_projection: float = 1.0
    preserve_anchor_tke: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SpectralPosteriorConfig:
        values = dict(values)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TKEMapConfig:
    """Configuration for explicit future-TKE inverse calibration."""

    base: dict[str, Any]
    width: int = 48
    depth: int = 5
    spectral_rank: int = 12
    modes_y: int = 10
    modes_x: int = 18
    expansion: int = 2
    dropout: float = 0.02
    maximum_log_ratio: float = 1.5
    correction_scale: float = 1.0
    interval_width_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TKEMapConfig:
        values = dict(values)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalTransferConfig:
    """Configuration for a local temporal-frequency inverse operator."""

    base: dict[str, Any]
    width: int = 48
    depth: int = 5
    spectral_rank: int = 12
    modes_y: int = 10
    modes_x: int = 18
    expansion: int = 2
    dropout: float = 0.02
    frequency_bins: int = 9
    maximum_log_amplitude: float = 0.7
    maximum_phase_shift: float = 1.0471975512
    correction_scale: float = 1.0
    phase_scale: float | None = None
    interval_width_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TemporalTransferConfig:
        values = dict(values)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankSpectralMix2d(nn.Module):
    """Global mixing at O(rank * H * W log(HW)), not O(width² * modes)."""

    def __init__(self, width: int, rank: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.rank = rank
        self.modes_y = modes_y
        self.modes_x = modes_x
        self.to_rank = nn.Conv2d(width, rank, 1, bias=False)
        self.from_rank = nn.Conv2d(rank, width, 1, bias=False)
        scale = 1.0 / max(1, rank)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(rank, modes_y, modes_x, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(rank, modes_y, modes_x, dtype=torch.cfloat)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.to_rank(features)
        height, width = projected.shape[-2:]
        spectrum = torch.fft.rfft2(projected.float(), norm="ortho")
        my = min(self.modes_y, height // 2)
        mx = min(self.modes_x, spectrum.shape[-1])
        filtered = torch.zeros_like(spectrum)
        filtered[..., :my, :mx] = spectrum[..., :my, :mx] * self.weight_pos[:, :my, :mx]
        filtered[..., -my:, :mx] = spectrum[..., -my:, :mx] * self.weight_neg[:, :my, :mx]
        mixed = torch.fft.irfft2(filtered, s=(height, width), norm="ortho")
        return self.from_rank(mixed.to(features.dtype))


class OperatorBlock(nn.Module):
    def __init__(
        self,
        width: int,
        rank: int,
        modes_y: int,
        modes_x: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.local = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.global_mix = LowRankSpectralMix2d(width, rank, modes_y, modes_x) if rank > 0 else None
        self.channel_mix = nn.Conv2d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        hidden = width * expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.residual_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(features)
        global_update = self.global_mix(normalized) if self.global_mix is not None else 0.0
        update = self.local(normalized) + global_update + self.channel_mix(normalized)
        features = features + self.residual_scale * update
        return features + self.residual_scale * self.ffn(self.norm2(features))


class MultiScaleUNetContext(nn.Module):
    """Two-scale encoder/decoder context with a full-resolution skip."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.width
        groups = 8 if width % 8 == 0 else 1

        def downsample() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(width, width, 3, stride=2, padding=1),
                nn.GroupNorm(groups, width),
                nn.GELU(),
            )

        self.down1 = downsample()
        self.block1 = OperatorBlock(
            width,
            config.spectral_rank,
            config.modes_y,
            config.modes_x,
            config.expansion,
            config.dropout,
        )
        self.down2 = downsample()
        self.block2 = OperatorBlock(
            width,
            config.spectral_rank,
            max(2, config.modes_y // 2),
            max(2, config.modes_x // 2),
            config.expansion,
            config.dropout,
        )
        self.decode1 = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.decode0 = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.scale = nn.Parameter(torch.tensor(1e-2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded1 = self.block1(self.down1(features))
        encoded2 = self.block2(self.down2(encoded1))
        decoded1 = F.interpolate(
            encoded2, size=encoded1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decode1(torch.cat((decoded1, encoded1), dim=1))
        decoded0 = F.interpolate(
            decoded1, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        return features + self.scale * self.decode0(torch.cat((decoded0, features), dim=1))


class LatentFlowBlock(nn.Module):
    """Residual block operating on complex Fourier coefficients packed as channels."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.channel = nn.Conv2d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(latent)
        latent = latent + self.scale * (self.spatial(normalized) + self.channel(normalized))
        return latent + self.scale * self.ffn(self.norm2(latent))


class SpectralPosteriorBlock(nn.Module):
    """Local 3D mixing over temporal and spatial frequency axes."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.frequency_mix = nn.Conv3d(width, width, 3, padding=1, groups=width)
        self.channel_mix = nn.Conv3d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(latent)
        latent = latent + self.scale * (
            self.frequency_mix(normalized) + self.channel_mix(normalized)
        )
        return latent + self.scale * self.ffn(self.norm2(latent))


class HybridFlowOperator(nn.Module):
    """Fast direct 20→20 forecaster with local and spectral receptive fields."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        modalities = 3 if config.use_pressure else 2
        descriptor_channels = 4 * modalities
        # time mean/std (u,v), local TKE, recent trend (u,v), and final
        # fluctuation (u,v). These expose exactly the statistics used by the
        # TKE metric without manufacturing any extra training samples.
        physics_channels = 9 if config.use_physics_features else 0
        # Flattened time/modalities + x/y coordinates + global/final descriptors
        # + optional physics summaries.
        in_channels = config.in_steps * modalities + 2 + descriptor_channels + physics_channels
        self.lift = nn.Conv2d(in_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.multiscale = MultiScaleUNetContext(config)
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, config.out_steps * 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv2d(config.width, config.width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width // 2, config.out_steps * 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(config.out_steps, 2))
        self.interval_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight)
        nn.init.constant_(self.width_head[-1].bias, -2.5)

    @staticmethod
    def _coordinates(
        batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        yy = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def _features(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if flow.ndim != 5 or flow.shape[1] != self.config.in_steps or flow.shape[-1] != 3:
            raise ValueError(f"expected [B,{self.config.in_steps},H,W,3], got {tuple(flow.shape)}")
        uv = flow[..., :2]
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_uv = (uv - uv_mean) / uv_std
        selected = uv
        if self.config.use_pressure:
            pressure = flow[..., 2:3]
            p_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
            p_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            normalized = torch.cat((normalized_uv, (pressure - p_mean) / p_std), dim=-1)
            selected = flow
        else:
            normalized = normalized_uv

        batch, _, height, width, _ = flow.shape
        flat = normalized.permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        global_mean = selected.mean(dim=(1, 2, 3))
        global_std = selected.std(dim=(1, 2, 3))
        final_mean = selected[:, -1].mean(dim=(1, 2))
        final_std = selected[:, -1].std(dim=(1, 2))
        descriptors = torch.cat((global_mean, global_std, final_mean, final_std), dim=-1)
        descriptors = descriptors[:, :, None, None].expand(-1, -1, height, width)
        coords = self._coordinates(batch, height, width, flow.device, flow.dtype)
        feature_parts = [flat, coords, descriptors]
        if self.config.use_physics_features:
            temporal_mean = uv.mean(dim=1)
            fluctuation = uv - temporal_mean[:, None]
            variance = fluctuation.square().mean(dim=1)
            temporal_std = variance.clamp_min(1e-8).sqrt()
            local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
            recent_index = max(0, self.config.in_steps - 5)
            recent_trend = (uv[:, -1] - uv[:, recent_index]) / max(
                1, self.config.in_steps - 1 - recent_index
            )
            final_fluctuation = uv[:, -1] - temporal_mean
            physics = torch.cat(
                (temporal_mean, temporal_std, local_tke, recent_trend, final_fluctuation),
                dim=-1,
            ).permute(0, 3, 1, 2)
            feature_parts.append(physics)
        features = torch.cat(feature_parts, dim=1)
        return features, uv_mean, uv_std

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raw = flow
        features, uv_mean, uv_std = self._features(raw)
        hidden = self.lift(features)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.blocks) // 2):
                hidden = self.multiscale(hidden)
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        )
        correction = correction.permute(0, 1, 3, 4, 2)
        base_normalized = (raw[:, -1:, ..., :2] - uv_mean) / uv_std
        recent_trend = (raw[:, -1:, ..., :2] - raw[:, -5:-4, ..., :2]) / (4 * uv_std)
        trend = recent_trend * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base_normalized + trend + correction) * uv_std + uv_mean
        pressure = torch.zeros_like(prediction_uv[..., :1])
        prediction = torch.cat((prediction_uv, pressure), dim=-1)

        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(batch, self.config.out_steps, 2, height, width)
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        half_width_uv = F.softplus(raw_width) * uv_std * self.interval_temperature.clamp_min(1e-3)
        half_width = torch.cat((half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1)
        return prediction, half_width


class DistilledUncertaintyOperator(nn.Module):
    """Fast interval student distilled from generative and domain-shift teachers.

    The deterministic point forecast remains the independently trained
    :class:`HybridFlowOperator`.  Only the interval morphology is learned here.
    The probability-flow teacher and sim/real disagreement are training-only, so
    deployment requires one anchor pass plus this small local head—no sampling.
    """

    def __init__(self, config: DistilledIntervalConfig) -> None:
        super().__init__()
        self.config = config
        anchor_config = ModelConfig.from_dict(config.anchor)
        self.anchor = HybridFlowOperator(anchor_config)
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.out_steps = anchor_config.out_steps
        # 9 input summaries + 7 forecast summaries + 2 coordinates + 8
        # sample-global descriptors. All velocity-valued features are normalized.
        diagnostic_channels = 26 + (2 * anchor_config.in_steps * 2 if config.phase_channels else 0)
        self.lift = nn.Conv2d(diagnostic_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    0,
                    1,
                    1,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.width_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, self.out_steps * 2, 1),
        )
        self.interval_temperature = nn.Parameter(
            torch.tensor(float(config.interval_scale)), requires_grad=False
        )
        nn.init.zeros_(self.width_head[-1].weight)
        nn.init.constant_(self.width_head[-1].bias, -2.5)

    def train(self, mode: bool = True) -> DistilledUncertaintyOperator:
        super().train(mode)
        self.anchor.eval()
        return self

    @staticmethod
    def _local_summaries(flow: torch.Tensor) -> torch.Tensor:
        temporal_mean = flow.mean(dim=1)
        fluctuation = flow - temporal_mean[:, None]
        variance = fluctuation.square().mean(dim=1)
        temporal_std = variance.clamp_min(1e-8).sqrt()
        local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
        recent = (flow[:, -1] - flow[:, -5]) / 4.0
        final_fluctuation = flow[:, -1] - temporal_mean
        return torch.cat(
            (temporal_mean, temporal_std, local_tke, recent, final_fluctuation),
            dim=-1,
        )

    def _diagnostics(
        self, flow: torch.Tensor, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uv = flow[..., :2]
        point = prediction[..., :2]
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_input = (uv - mean) / std
        normalized_point = (point - mean) / std

        input_summary = self._local_summaries(normalized_input)
        point_mean = normalized_point.mean(dim=1)
        point_fluctuation = normalized_point - point_mean[:, None]
        point_variance = point_fluctuation.square().mean(dim=1)
        point_summary = torch.cat(
            (
                point_mean,
                point_variance.clamp_min(1e-8).sqrt(),
                0.5 * point_variance.sum(dim=-1, keepdim=True),
                normalized_point[:, -1] - normalized_input[:, -1],
            ),
            dim=-1,
        )
        batch, _, height, width, _ = flow.shape
        coords = HybridFlowOperator._coordinates(
            batch, height, width, flow.device, flow.dtype
        ).permute(0, 2, 3, 1)
        descriptors = torch.cat(
            (
                normalized_input[:, -1].mean(dim=(1, 2)),
                normalized_input[:, -1].std(dim=(1, 2)),
                normalized_point[:, -1].mean(dim=(1, 2)),
                normalized_point.std(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        descriptors = descriptors[:, None, None].expand(-1, height, width, -1)
        diagnostics = torch.cat((input_summary, point_summary, coords, descriptors), dim=-1)
        if self.config.phase_channels:
            input_phase = normalized_input.permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
            point_phase = normalized_point.permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
            diagnostics = torch.cat((diagnostics, input_phase, point_phase), dim=-1)
        diagnostics = diagnostics.permute(0, 3, 1, 2)
        return diagnostics, std

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prediction = self.anchor(flow)
        if not return_interval:
            return prediction
        diagnostics, std = self._diagnostics(flow, prediction)
        hidden = self.lift(diagnostics)
        for block in self.blocks:
            hidden = block(hidden)
        raw_width = self.width_head(self.final_norm(hidden)).reshape(
            flow.shape[0], self.out_steps, 2, flow.shape[2], flow.shape[3]
        )
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        half_width_uv = F.softplus(raw_width) * std * self.interval_temperature.clamp_min(1e-3)
        half_width = torch.cat((half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1)
        return prediction, half_width


class PhaseConditionedLatentFlow(nn.Module):
    """Conditional probability flow over phase-preserving Fourier residuals."""

    def __init__(self, config: LatentFlowConfig) -> None:
        super().__init__()
        self.config = config
        teacher_config = ModelConfig.from_dict(config.teacher)
        self.teacher = HybridFlowOperator(teacher_config)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.out_steps = teacher_config.out_steps
        self.latent_channels = self.out_steps * 2 * 2
        self.latent_height = 2 * config.modes_y
        self.latent_width = config.modes_x
        condition_channels = self.latent_channels + 2 + 5
        self.lift = nn.Conv2d(self.latent_channels + condition_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                LatentFlowBlock(config.width, config.expansion, config.dropout)
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, self.latent_channels, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        yy = torch.linspace(-1.0, 1.0, self.latent_height)
        xx = torch.linspace(0.0, 1.0, self.latent_width)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        self.register_buffer(
            "frequency_coordinates",
            torch.stack((grid_x, grid_y), dim=0),
            persistent=True,
        )
        generator = torch.Generator().manual_seed(config.sample_seed)
        noise = torch.randn(
            config.max_ensemble_size,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            generator=generator,
        )
        for index in range(1, config.max_ensemble_size, 2):
            noise[index].copy_(-noise[index - 1])
        self.register_buffer("base_noise", noise, persistent=True)

    def train(self, mode: bool = True) -> PhaseConditionedLatentFlow:
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def _statistics(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = flow[..., :2]
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return mean, std

    def pack_fourier(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 5 or field.shape[1] != self.out_steps or field.shape[-1] != 2:
            raise ValueError(f"expected [B,{self.out_steps},H,W,2], got {tuple(field.shape)}")
        height, width = field.shape[2:4]
        if 2 * self.config.modes_y > height:
            raise ValueError("twice modes_y must not exceed the physical height")
        if self.config.modes_x > width // 2 + 1:
            raise ValueError("modes_x exceeds the real FFT width")
        spectrum = torch.fft.rfft2(field.float(), dim=(2, 3), norm="ortho")
        selected = torch.cat(
            (
                spectrum[:, :, : self.config.modes_y, : self.config.modes_x],
                spectrum[:, :, -self.config.modes_y :, : self.config.modes_x],
            ),
            dim=2,
        )
        packed = torch.view_as_real(selected)
        packed = packed.permute(0, 1, 4, 5, 2, 3).contiguous()
        return packed.reshape(
            field.shape[0], self.latent_channels, self.latent_height, self.latent_width
        )

    def unpack_fourier(
        self, latent: torch.Tensor, *, height: int = 32, width: int = 64
    ) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        ):
            raise ValueError(f"unexpected latent shape {tuple(latent.shape)}")
        pairs = latent.reshape(
            latent.shape[0],
            self.out_steps,
            2,
            2,
            self.latent_height,
            self.latent_width,
        )
        pairs = pairs.permute(0, 1, 4, 5, 2, 3).contiguous()
        selected = torch.view_as_complex(pairs.float())
        spectrum = torch.zeros(
            latent.shape[0],
            self.out_steps,
            height,
            width // 2 + 1,
            2,
            dtype=torch.cfloat,
            device=latent.device,
        )
        modes_y = self.config.modes_y
        modes_x = self.config.modes_x
        spectrum[:, :, :modes_y, :modes_x] = selected[:, :, :modes_y]
        spectrum[:, :, -modes_y:, :modes_x] = selected[:, :, modes_y:]
        return torch.fft.irfft2(spectrum, s=(height, width), dim=(2, 3), norm="ortho")

    def _condition(
        self, anchor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        normalized_anchor = (anchor[..., :2] - mean) / std
        anchor_latent = self.pack_fourier(normalized_anchor)
        coordinates = self.frequency_coordinates.to(anchor_latent.dtype)
        coordinates = coordinates.unsqueeze(0).expand(anchor.shape[0], -1, -1, -1)
        return torch.cat((anchor_latent, coordinates), dim=1)

    @staticmethod
    def _time_features(time_value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        time_value = time_value.reshape(-1, 1, 1, 1)
        features = torch.cat(
            (
                time_value,
                torch.sin(torch.pi * time_value),
                torch.cos(torch.pi * time_value),
                torch.sin(2.0 * torch.pi * time_value),
                torch.cos(2.0 * torch.pi * time_value),
            ),
            dim=1,
        )
        return features.expand(-1, -1, height, width)

    def velocity(
        self, latent: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        time_features = self._time_features(time_value, latent.shape[-2], latent.shape[-1]).to(
            latent.dtype
        )
        hidden = self.lift(torch.cat((latent, condition, time_features), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.velocity_head(self.final_norm(hidden))

    def flow_matching_prediction(
        self, flow: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            anchor = self.teacher(flow)
        mean, std = self._statistics(flow)
        residual = (target[..., :2] - anchor[..., :2]) / std
        target_latent = self.pack_fourier(residual) * self.config.data_scale
        noise = torch.randn_like(target_latent)
        time_value = torch.rand(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        shaped_time = time_value[:, None, None, None]
        interpolated = (1.0 - shaped_time) * noise + shaped_time * target_latent
        condition = self._condition(anchor, mean, std)
        velocity = self.velocity(interpolated, time_value, condition)
        velocity_target = target_latent - noise
        endpoint_latent = interpolated + (1.0 - shaped_time) * velocity
        endpoint_residual = self.unpack_fourier(
            endpoint_latent, height=flow.shape[2], width=flow.shape[3]
        )
        prediction_uv = anchor[..., :2] + endpoint_residual * std / self.config.data_scale
        pressure = torch.zeros_like(prediction_uv[..., :1])
        prediction = torch.cat((prediction_uv, pressure), dim=-1)
        return prediction, {
            "flow": (velocity - velocity_target).square().mean(),
            "phase": self.phase_loss(endpoint_latent, target_latent),
        }

    def phase_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.reshape(
            prediction.shape[0], self.out_steps, 2, 2, *prediction.shape[-2:]
        )
        target = target.reshape(target.shape[0], self.out_steps, 2, 2, *target.shape[-2:])
        dot = (prediction * target).sum(dim=3)
        pred_norm = prediction.square().sum(dim=3).clamp_min(1e-8).sqrt()
        target_norm = target.square().sum(dim=3).clamp_min(1e-8).sqrt()
        alignment = dot / (pred_norm * target_norm).clamp_min(1e-8)
        weight = target_norm / target_norm.mean().clamp_min(1e-8)
        return ((1.0 - alignment) * weight).mean()

    def _integrate(
        self,
        initial: torch.Tensor,
        condition: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        latent = initial
        step_size = 1.0 / steps
        for index in range(steps):
            time_value = latent.new_full((latent.shape[0],), index / steps)
            latent = latent + step_size * self.velocity(latent, time_value, condition)
        return latent

    def sample(
        self,
        flow: torch.Tensor,
        *,
        ensemble_size: int | None = None,
        sample_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, teacher_width = self.teacher(flow, return_interval=True)
        mean, std = self._statistics(flow)
        condition = self._condition(anchor, mean, std)
        batch = flow.shape[0]
        ensemble = ensemble_size or self.config.ensemble_size
        steps = sample_steps or self.config.sample_steps
        if not 1 <= ensemble <= self.config.max_ensemble_size:
            raise ValueError("ensemble_size exceeds the configured deterministic noise bank")
        if steps < 1:
            raise ValueError("sample_steps must be positive")
        initial = self.base_noise[:ensemble].unsqueeze(0).expand(batch, -1, -1, -1, -1)
        initial = initial.reshape(
            batch * ensemble,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        repeated_condition = condition[:, None].expand(-1, ensemble, -1, -1, -1)
        repeated_condition = repeated_condition.reshape(
            batch * ensemble, condition.shape[1], self.latent_height, self.latent_width
        )
        sampled_latent = self._integrate(initial, repeated_condition, steps)
        residual = self.unpack_fourier(sampled_latent, height=flow.shape[2], width=flow.shape[3])
        residual = residual.reshape(batch, ensemble, self.out_steps, *flow.shape[2:4], 2)
        samples = anchor[:, None, ..., :2] + residual * std[:, None] / self.config.data_scale

        if self.config.point_mode == "anchor":
            point_uv = anchor[..., :2]
        elif self.config.point_mode == "mode":
            mode_latent = torch.zeros(
                batch,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
                device=flow.device,
                dtype=sampled_latent.dtype,
            )
            mode_latent = self._integrate(mode_latent, condition, steps)
            mode_residual = self.unpack_fourier(
                mode_latent, height=flow.shape[2], width=flow.shape[3]
            )
            point_uv = anchor[..., :2] + mode_residual * std / self.config.data_scale
        elif self.config.point_mode == "mean":
            point_uv = samples.mean(dim=1)
        elif self.config.point_mode == "medoid":
            center = samples.mean(dim=1, keepdim=True)
            distance = (samples - center).square().mean(dim=(2, 3, 4, 5))
            indices = distance.argmin(dim=1)
            point_uv = samples[torch.arange(batch, device=flow.device), indices]
        else:
            raise ValueError(f"unknown point_mode: {self.config.point_mode}")

        spread = (samples - point_uv[:, None]).square().mean(dim=1).clamp_min(1e-10).sqrt()
        half_width_uv = spread * self.config.interval_scale
        if self.config.teacher_width_floor > 0.0:
            half_width_uv = torch.maximum(
                half_width_uv,
                teacher_width[..., :2] * self.config.teacher_width_floor,
            )
        pressure = torch.zeros_like(point_uv[..., :1])
        prediction = torch.cat((point_uv, pressure), dim=-1)
        half_width = torch.cat((half_width_uv, torch.zeros_like(pressure)), dim=-1)
        return prediction, half_width, samples

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        prediction, half_width, _ = self.sample(flow)
        if return_interval:
            return prediction, half_width
        return prediction


class MaskedSpectralPosteriorRefiner(nn.Module):
    """Infer the missing future of a joint trajectory as a spectral inverse problem.

    The first half of the 40-frame trajectory is observed. Every reverse-flow
    update is projected onto that measurement set in physical space, while the
    learned transport operates on whitened complex 3D Fourier coefficients.
    """

    def __init__(self, config: SpectralPosteriorConfig) -> None:
        super().__init__()
        self.config = config
        anchor_config = ModelConfig.from_dict(config.anchor)
        self.anchor = HybridFlowOperator(anchor_config)
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        if config.interval_teacher is None:
            self.interval_teacher = None
        else:
            self.interval_teacher = DistilledUncertaintyOperator(
                DistilledIntervalConfig.from_dict(config.interval_teacher)
            )
            self.interval_teacher.requires_grad_(False)
            self.interval_teacher.eval()
        self.in_steps = anchor_config.in_steps
        self.out_steps = anchor_config.out_steps
        self.trajectory_steps = self.in_steps + self.out_steps
        if 2 * config.modes_t > self.trajectory_steps:
            raise ValueError("twice modes_t exceeds the joint trajectory length")
        if 2 * config.modes_y > 32:
            raise ValueError("twice modes_y exceeds the physical height")
        if config.modes_x > 33:
            raise ValueError("modes_x exceeds the real FFT width")

        self.latent_channels = 4  # two velocities x real/imaginary
        self.latent_shape = (2 * config.modes_t, 2 * config.modes_y, config.modes_x)
        condition_channels = self.latent_channels + 3 + 6
        self.lift = nn.Conv3d(
            self.latent_channels + condition_channels + 5,
            config.width,
            1,
        )
        self.blocks = nn.ModuleList(
            [
                SpectralPosteriorBlock(config.width, config.expansion, config.dropout)
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv3d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(config.width, self.latent_channels, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        frequency_t = torch.linspace(-1.0, 1.0, self.latent_shape[0])
        frequency_y = torch.linspace(-1.0, 1.0, self.latent_shape[1])
        frequency_x = torch.linspace(0.0, 1.0, self.latent_shape[2])
        grid_t, grid_y, grid_x = torch.meshgrid(
            frequency_t, frequency_y, frequency_x, indexing="ij"
        )
        self.register_buffer(
            "frequency_coordinates",
            torch.stack((grid_t, grid_y, grid_x), dim=0),
            persistent=True,
        )
        self.register_buffer(
            "spectral_scale",
            torch.ones(1, self.latent_channels, *self.latent_shape),
            persistent=True,
        )
        generator = torch.Generator().manual_seed(config.sample_seed)
        noise = torch.randn(
            config.max_ensemble_size,
            self.latent_channels,
            *self.latent_shape,
            generator=generator,
        )
        for index in range(1, config.max_ensemble_size, 2):
            noise[index].copy_(-noise[index - 1])
        self.register_buffer("base_noise", noise, persistent=True)

    def train(self, mode: bool = True) -> MaskedSpectralPosteriorRefiner:
        super().train(mode)
        self.anchor.eval()
        if self.interval_teacher is not None:
            self.interval_teacher.eval()
        return self

    def set_spectral_scale(self, scale: torch.Tensor, minimum_ratio: float = 0.05) -> None:
        if scale.shape != self.spectral_scale.shape:
            raise ValueError(
                f"spectral scale shape {tuple(scale.shape)} != {tuple(self.spectral_scale.shape)}"
            )
        finite = scale.detach().to(self.spectral_scale).clamp_min(0.0)
        positive = finite[finite > 0.0]
        if positive.numel() == 0:
            raise ValueError("spectral scale contains no positive coefficient")
        floor = positive.median() * float(minimum_ratio)
        self.spectral_scale.copy_(finite.clamp_min(floor).clamp_max(positive.max() * 10.0))

    @staticmethod
    def _statistics(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = flow[..., :2]
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return mean, std

    def _joint_base(self, flow: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return torch.cat((flow[..., :2], anchor[..., :2]), dim=1)

    def pack_fourier(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.trajectory_steps, 32, 64, 2)
        if field.ndim != 5 or tuple(field.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(field.shape)}")
        spectrum = torch.fft.rfftn(field.float(), dim=(1, 2, 3), norm="ortho")
        mt = self.config.modes_t
        my = self.config.modes_y
        mx = self.config.modes_x
        selected_t = torch.cat((spectrum[:, :mt], spectrum[:, -mt:]), dim=1)
        selected = torch.cat((selected_t[:, :, :my], selected_t[:, :, -my:]), dim=2)
        selected = selected[:, :, :, :mx]
        packed = torch.view_as_real(selected)
        packed = packed.permute(0, 4, 5, 1, 2, 3).contiguous()
        return packed.reshape(field.shape[0], self.latent_channels, *self.latent_shape)

    def unpack_fourier(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_channels, *self.latent_shape)
        if latent.ndim != 5 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(latent.shape)}")
        pairs = latent.reshape(latent.shape[0], 2, 2, *self.latent_shape)
        pairs = pairs.permute(0, 3, 4, 5, 1, 2).contiguous()
        selected = torch.view_as_complex(pairs.float())
        spectrum = torch.zeros(
            latent.shape[0],
            self.trajectory_steps,
            32,
            33,
            2,
            dtype=torch.cfloat,
            device=latent.device,
        )
        mt = self.config.modes_t
        my = self.config.modes_y
        mx = self.config.modes_x
        spectrum[:, :mt, :my, :mx] = selected[:, :mt, :my]
        spectrum[:, :mt, -my:, :mx] = selected[:, :mt, my:]
        spectrum[:, -mt:, :my, :mx] = selected[:, mt:, :my]
        spectrum[:, -mt:, -my:, :mx] = selected[:, mt:, my:]

        # The k_x=0 plane of an rFFT must be Hermitian over (time, y).  Our
        # rectangular low-frequency crop can contain one side of a conjugate
        # pair without the other (for example k_t=-modes_t).  Complete those
        # missing partners before irfftn; when both sides are represented,
        # average them onto the valid real-signal subspace.  This keeps
        # pack -> unpack -> pack closed while retaining the DC/low-x modes.
        selected_mask = torch.zeros(
            self.trajectory_steps, 32, dtype=torch.bool, device=latent.device
        )
        selected_mask[:mt, :my] = True
        selected_mask[:mt, -my:] = True
        selected_mask[-mt:, :my] = True
        selected_mask[-mt:, -my:] = True
        negative_t = (-torch.arange(self.trajectory_steps, device=latent.device)) % (
            self.trajectory_steps
        )
        negative_y = (-torch.arange(32, device=latent.device)) % 32
        plane = spectrum[:, :, :, 0]
        mirrored = plane.index_select(1, negative_t).index_select(2, negative_y).conj()
        mirrored_mask = selected_mask.index_select(0, negative_t).index_select(1, negative_y)
        both = (selected_mask & mirrored_mask)[None, :, :, None]
        only_mirror = ((~selected_mask) & mirrored_mask)[None, :, :, None]
        plane = torch.where(both, 0.5 * (plane + mirrored), plane)
        plane = torch.where(only_mirror, mirrored, plane)
        spectrum[:, :, :, 0] = plane
        return torch.fft.irfftn(
            spectrum,
            s=(self.trajectory_steps, 32, 64),
            dim=(1, 2, 3),
            norm="ortho",
        )

    def encode_residual(self, residual: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        normalized = residual / std
        return self.pack_fourier(normalized) / self.spectral_scale.clamp_min(1e-6)

    def decode_residual(self, latent: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        normalized = self.unpack_fourier(latent * self.spectral_scale)
        return normalized * std

    def project_observation(self, latent: torch.Tensor) -> torch.Tensor:
        strength = float(self.config.observation_projection)
        if strength <= 0.0:
            return latent
        normalized = self.unpack_fourier(latent * self.spectral_scale)
        projected = normalized.clone()
        projected[:, : self.in_steps] = 0.0
        encoded = self.pack_fourier(projected) / self.spectral_scale.clamp_min(1e-6)
        return torch.lerp(latent, encoded, min(1.0, strength))

    @staticmethod
    def _diffusion_time_features(
        time_value: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        time_value = time_value.reshape(-1, 1, 1, 1, 1)
        features = torch.cat(
            (
                time_value,
                torch.sin(torch.pi * time_value),
                torch.cos(torch.pi * time_value),
                torch.sin(2.0 * torch.pi * time_value),
                torch.cos(2.0 * torch.pi * time_value),
            ),
            dim=1,
        )
        return features.expand(-1, -1, *shape)

    def _condition(
        self,
        flow: torch.Tensor,
        anchor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        joint = (self._joint_base(flow, anchor) - mean) / std
        base_latent = self.pack_fourier(joint)
        base_latent = torch.sign(base_latent) * torch.log1p(base_latent.abs())
        coordinates = self.frequency_coordinates.to(base_latent.dtype)
        coordinates = coordinates.unsqueeze(0).expand(flow.shape[0], -1, -1, -1, -1)
        uv = flow[..., :2]
        descriptors = torch.cat(
            (
                uv.mean(dim=(1, 2, 3)),
                uv.std(dim=(1, 2, 3)),
                (uv[:, -1] - uv[:, -5]).mean(dim=(1, 2)) / 4.0,
            ),
            dim=-1,
        )
        descriptors = descriptors[:, :, None, None, None].expand(-1, -1, *self.latent_shape)
        return torch.cat((base_latent, coordinates, descriptors), dim=1)

    def velocity(
        self, latent: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        time_features = self._diffusion_time_features(time_value, self.latent_shape)
        hidden = self.lift(torch.cat((latent, condition, time_features), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.velocity_head(self.final_norm(hidden))

    def _target_residual(self, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        past = torch.zeros(
            target.shape[0],
            self.in_steps,
            target.shape[2],
            target.shape[3],
            2,
            device=target.device,
            dtype=target.dtype,
        )
        future = target[..., :2] - anchor[..., :2]
        return torch.cat((past, future), dim=1)

    @staticmethod
    def tke_preserving_point(
        anchor: torch.Tensor,
        correction: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Move on the local temporal-energy sphere instead of changing TKE.

        At every spatial location, remove the radial component of the
        zero-mean correction and retract the remaining tangent update onto the
        anchor's temporal fluctuation-energy sphere. The temporal mean remains
        free, so mean wake/profile errors can still improve while the exact TKE
        map used by the scorer is preserved (up to floating-point roundoff).
        """
        anchor_uv = anchor[..., :2].float()
        correction_uv = correction.float()
        anchor_mean = anchor_uv.mean(dim=1, keepdim=True)
        correction_mean = correction_uv.mean(dim=1, keepdim=True)
        anchor_fluctuation = anchor_uv - anchor_mean
        correction_fluctuation = correction_uv - correction_mean
        radial = (anchor_fluctuation * correction_fluctuation).sum(
            dim=(1, 4), keepdim=True
        )
        anchor_energy = anchor_fluctuation.square().sum(dim=(1, 4), keepdim=True)
        tangent = correction_fluctuation - anchor_fluctuation * (
            radial / anchor_energy.clamp_min(1e-12)
        )
        candidate = anchor_fluctuation + float(scale) * tangent
        candidate_energy = candidate.square().sum(dim=(1, 4), keepdim=True)
        retraction = (anchor_energy / candidate_energy.clamp_min(1e-12)).sqrt()
        retraction = torch.where(anchor_energy > 1e-12, retraction, torch.zeros_like(retraction))
        point_mean = anchor_mean + float(scale) * correction_mean
        return point_mean + candidate * retraction

    @staticmethod
    def temporal_frequency_correction(
        correction: torch.Tensor,
        gains: tuple[float, float, float, float] | list[float],
    ) -> torch.Tensor:
        """Apply independent mean/low/mid/high gains without discarding a mode."""
        if len(gains) != 4:
            raise ValueError("frequency gains must be [mean, low, mid, high]")
        mean = correction.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(correction - mean, dim=1, norm="ortho")
        if spectrum.shape[1] != 11:
            raise ValueError("frequency-banded tangent expects 20 output steps")
        frequency_gain = spectrum.real.new_zeros(11)
        frequency_gain[1:3] = float(gains[1])
        frequency_gain[3:6] = float(gains[2])
        frequency_gain[6:] = float(gains[3])
        fluctuation = torch.fft.irfft(
            spectrum * frequency_gain.view(1, -1, 1, 1, 1),
            n=correction.shape[1],
            dim=1,
            norm="ortho",
        )
        return float(gains[0]) * mean + fluctuation

    @staticmethod
    def spatial_frequency_correction(
        correction: torch.Tensor,
        *,
        cutoff_y: int,
        cutoff_x: int,
        low_gain: float,
        high_gain: float,
    ) -> torch.Tensor:
        """Reweight resolved and sub-filter spatial modes without phase loss."""
        height, width = correction.shape[2:4]
        spectrum = torch.fft.rfft2(correction.float(), dim=(2, 3), norm="ortho")
        y = torch.arange(height, device=correction.device)
        signed_y = torch.minimum(y, height - y)
        x = torch.arange(width // 2 + 1, device=correction.device)
        low_mask = (signed_y[:, None] <= int(cutoff_y)) & (
            x[None, :] <= int(cutoff_x)
        )
        gain = torch.where(
            low_mask,
            spectrum.real.new_tensor(float(low_gain)),
            spectrum.real.new_tensor(float(high_gain)),
        )
        return torch.fft.irfft2(
            spectrum * gain[None, None, :, :, None],
            s=(height, width),
            dim=(2, 3),
            norm="ortho",
        )

    @classmethod
    def tke_preserving_frequency_point(
        cls,
        anchor: torch.Tensor,
        correction: torch.Tensor,
        gains: tuple[float, float, float, float] | list[float],
    ) -> torch.Tensor:
        filtered = cls.temporal_frequency_correction(correction, gains)
        return cls.tke_preserving_point(anchor, filtered, 1.0)

    @classmethod
    def tke_preserving_spatiotemporal_frequency_point(
        cls,
        anchor: torch.Tensor,
        correction: torch.Tensor,
        temporal_gains: tuple[float, float, float, float] | list[float],
        spatial: dict[str, Any],
    ) -> torch.Tensor:
        filtered = cls.temporal_frequency_correction(correction, temporal_gains)
        filtered = cls.spatial_frequency_correction(
            filtered,
            cutoff_y=int(spatial["cutoff_y"]),
            cutoff_x=int(spatial["cutoff_x"]),
            low_gain=float(spatial["low_gain"]),
            high_gain=float(spatial["high_gain"]),
        )
        return cls.tke_preserving_point(anchor, filtered, 1.0)

    def corrected_point(
        self,
        anchor: torch.Tensor,
        correction: torch.Tensor,
        scale: float,
        *,
        preserve_anchor_tke: bool | None = None,
    ) -> torch.Tensor:
        preserve = (
            self.config.preserve_anchor_tke
            if preserve_anchor_tke is None
            else bool(preserve_anchor_tke)
        )
        if preserve:
            return self.tke_preserving_point(anchor, correction, scale)
        return anchor[..., :2] + float(scale) * correction

    def flow_matching_prediction(
        self, flow: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            anchor = self.anchor(flow)
        mean, std = self._statistics(flow)
        target_latent = self.encode_residual(self._target_residual(anchor, target), std)
        noise = self.project_observation(torch.randn_like(target_latent))
        time_value = torch.rand(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        shaped_time = time_value[:, None, None, None, None]
        interpolated = (1.0 - shaped_time) * noise + shaped_time * target_latent
        condition = self._condition(flow, anchor, mean, std)
        velocity = self.velocity(interpolated, time_value, condition)
        velocity_target = target_latent - noise
        endpoint = self.project_observation(interpolated + (1.0 - shaped_time) * velocity)
        residual = self.decode_residual(endpoint, std)[:, self.in_steps :]
        prediction_uv = anchor[..., :2] + residual
        prediction = torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)
        return prediction, {
            "flow": (velocity - velocity_target).square().mean(),
            "endpoint": (endpoint - target_latent).square().mean(),
        }

    def mode_prediction(
        self,
        flow: torch.Tensor,
        *,
        correction_scale: float = 1.0,
        steps: int | None = None,
        preserve_anchor_tke: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiate the exact zero-initialized point-inference path.

        Flow matching trains velocities at random interpolated latents, while
        Track 1 point inference uses the deterministic posterior mode that
        starts at zero. Keeping this path explicit lets a second training
        stage optimize the deployed estimator directly without unfreezing the
        anchor or invoking the interval/ensemble branches.
        """
        with torch.no_grad():
            anchor = self.anchor(flow)
        mean, std = self._statistics(flow)
        condition = self._condition(flow, anchor, mean, std)
        initial = torch.zeros(
            flow.shape[0],
            self.latent_channels,
            *self.latent_shape,
            device=flow.device,
            dtype=condition.dtype,
        )
        integration_steps = self.config.sample_steps if steps is None else int(steps)
        if integration_steps < 1:
            raise ValueError("mode integration steps must be positive")
        mode_latent = self._integrate(initial, condition, integration_steps)
        correction = self.decode_residual(mode_latent, std)[:, self.in_steps :]
        point_uv = self.corrected_point(
            anchor,
            correction,
            correction_scale,
            preserve_anchor_tke=preserve_anchor_tke,
        )
        prediction = torch.cat((point_uv, torch.zeros_like(point_uv[..., :1])), dim=-1)
        return prediction, correction, anchor

    def _integrate(
        self, initial: torch.Tensor, condition: torch.Tensor, steps: int
    ) -> torch.Tensor:
        latent = self.project_observation(initial)
        step_size = 1.0 / steps
        for index in range(steps):
            time_value = latent.new_full((latent.shape[0],), index / steps)
            latent = latent + step_size * self.velocity(latent, time_value, condition)
            latent = self.project_observation(latent)
        return latent

    def _anchor_and_width(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor, anchor_width = self.anchor(flow, return_interval=True)
        if self.interval_teacher is not None:
            _, anchor_width = self.interval_teacher(flow, return_interval=True)
        return anchor, anchor_width

    def posterior_components(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, anchor_width = self._anchor_and_width(flow)
        mean, std = self._statistics(flow)
        condition = self._condition(flow, anchor, mean, std)
        batch = flow.shape[0]
        ensemble = self.config.ensemble_size
        steps = self.config.sample_steps
        if not 1 <= ensemble <= self.config.max_ensemble_size:
            raise ValueError("ensemble_size exceeds deterministic noise bank")
        if steps < 1:
            raise ValueError("sample_steps must be positive")
        initial = self.base_noise[:ensemble].unsqueeze(0).expand(batch, -1, -1, -1, -1, -1)
        initial = initial.reshape(batch * ensemble, self.latent_channels, *self.latent_shape)
        repeated_condition = condition[:, None].expand(-1, ensemble, -1, -1, -1, -1)
        repeated_condition = repeated_condition.reshape(
            batch * ensemble, condition.shape[1], *self.latent_shape
        )
        sampled = self._integrate(initial, repeated_condition, steps)
        repeated_std = std[:, None].expand(-1, ensemble, -1, -1, -1, -1)
        repeated_std = repeated_std.reshape(batch * ensemble, 1, 1, 1, 2)
        residual = self.decode_residual(sampled, repeated_std)
        residual = residual[:, self.in_steps :].reshape(batch, ensemble, self.out_steps, 32, 64, 2)
        mean_residual = residual.mean(dim=1)

        mode_initial = torch.zeros(
            batch,
            self.latent_channels,
            *self.latent_shape,
            device=flow.device,
            dtype=sampled.dtype,
        )
        mode_latent = self._integrate(mode_initial, condition, steps)
        mode_residual = self.decode_residual(mode_latent, std)[:, self.in_steps :]
        spread = (residual - mean_residual[:, None]).square().mean(dim=1).clamp_min(1e-10).sqrt()
        return anchor, mean_residual, mode_residual, spread, anchor_width

    def sample(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.interval_scale == 0.0 and (
            self.config.correction_source == "mode" or self.config.point_correction_scale == 0.0
        ):
            anchor, anchor_width = self._anchor_and_width(flow)
            if self.config.point_correction_scale == 0.0:
                correction = torch.zeros_like(anchor[..., :2])
            else:
                mean, std = self._statistics(flow)
                condition = self._condition(flow, anchor, mean, std)
                initial = torch.zeros(
                    flow.shape[0],
                    self.latent_channels,
                    *self.latent_shape,
                    device=flow.device,
                    dtype=condition.dtype,
                )
                mode_latent = self._integrate(initial, condition, self.config.sample_steps)
                correction = self.decode_residual(mode_latent, std)[:, self.in_steps :]
            point_uv = self.corrected_point(
                anchor, correction, float(self.config.point_correction_scale)
            )
            half_width_uv = anchor_width[..., :2] * float(self.config.anchor_width_floor)
            prediction = torch.cat((point_uv, torch.zeros_like(point_uv[..., :1])), dim=-1)
            half_width = torch.cat(
                (half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1
            )
            return prediction, half_width, correction

        anchor, mean_residual, mode_residual, spread, anchor_width = self.posterior_components(flow)
        if self.config.correction_source == "mean":
            correction = mean_residual
        elif self.config.correction_source == "mode":
            correction = mode_residual
        else:
            raise ValueError(f"unknown correction_source: {self.config.correction_source}")
        point_uv = self.corrected_point(
            anchor, correction, float(self.config.point_correction_scale)
        )
        half_width_uv = spread * float(self.config.interval_scale)
        if self.config.anchor_width_floor > 0.0:
            half_width_uv = torch.maximum(
                half_width_uv,
                anchor_width[..., :2] * float(self.config.anchor_width_floor),
            )
        prediction = torch.cat((point_uv, torch.zeros_like(point_uv[..., :1])), dim=-1)
        half_width = torch.cat((half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1)
        return prediction, half_width, correction

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        prediction, half_width, _ = self.sample(flow)
        if return_interval:
            return prediction, half_width
        return prediction


class TKEMapCorrector(nn.Module):
    """Estimate future TKE explicitly, then reproject a frozen trajectory.

    The frozen base retains its mean flow and temporal phase. A compact spatial
    neural operator observes the full measured sequence plus base summaries and
    predicts a bounded log-ratio between target and base TKE at every grid cell.
    Rescaling only temporal fluctuations realizes that TKE map exactly.
    """

    feature_channels = 56

    def __init__(self, config: TKEMapConfig) -> None:
        super().__init__()
        self.config = config
        self.base = MaskedSpectralPosteriorRefiner(
            SpectralPosteriorConfig.from_dict(config.base)
        )
        self.base.requires_grad_(False)
        self.base.eval()
        self.lift = nn.Conv2d(self.feature_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.ratio_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, 1, 1),
        )
        nn.init.zeros_(self.ratio_head[-1].weight)
        nn.init.zeros_(self.ratio_head[-1].bias)
        y = torch.linspace(-1.0, 1.0, 32)
        x = torch.linspace(-1.0, 1.0, 64)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer(
            "spatial_coordinates", torch.stack((grid_y, grid_x), dim=0), persistent=True
        )

    def train(self, mode: bool = True) -> TKEMapCorrector:
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def temporal_tke(velocity: torch.Tensor) -> torch.Tensor:
        uv = velocity[..., :2]
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1)

    def features(self, flow: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        uv = flow[..., :2]
        base_uv = base_prediction[..., :2]
        sequence = uv.permute(0, 1, 4, 2, 3).reshape(flow.shape[0], 40, 32, 64)
        observed_mean = uv.mean(dim=1)
        observed_std = uv.std(dim=1)
        observed_trend = uv[:, -1] - uv[:, 0]
        base_mean = base_uv.mean(dim=1)
        base_std = base_uv.std(dim=1)
        spatial = torch.cat(
            (observed_mean, observed_std, observed_trend, base_mean, base_std), dim=-1
        ).permute(0, 3, 1, 2)
        energy = torch.stack(
            (
                self.temporal_tke(uv),
                self.temporal_tke(uv[:, :10]),
                self.temporal_tke(uv[:, 10:]),
                self.temporal_tke(base_uv),
            ),
            dim=1,
        )
        energy = torch.sign(energy) * torch.log1p(energy.abs() / 1e-3)
        coordinates = self.spatial_coordinates.to(flow.dtype).unsqueeze(0)
        coordinates = coordinates.expand(flow.shape[0], -1, -1, -1)
        result = torch.cat((sequence, spatial, energy, coordinates), dim=1)
        if result.shape[1] != self.feature_channels:
            raise RuntimeError(f"unexpected TKE feature channels: {result.shape[1]}")
        return result

    def predict_log_ratio(
        self, flow: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.lift(self.features(flow, base_prediction))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.ratio_head(self.final_norm(hidden))
        maximum = float(self.config.maximum_log_ratio)
        return maximum * torch.tanh(raw / max(maximum, 1e-6))

    @staticmethod
    def apply_log_tke_ratio(
        prediction: torch.Tensor, log_ratio: torch.Tensor, scale: float
    ) -> torch.Tensor:
        uv = prediction[..., :2].float()
        mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - mean
        factor = torch.exp(0.5 * float(scale) * log_ratio).permute(0, 2, 3, 1)
        corrected = mean + fluctuation * factor[:, None]
        return torch.cat((corrected, torch.zeros_like(corrected[..., :1])), dim=-1)

    def components(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base_prediction, half_width = self.base(flow, return_interval=True)
        log_ratio = self.predict_log_ratio(flow, base_prediction)
        return base_prediction, half_width, log_ratio

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base_prediction, half_width, log_ratio = self.components(flow)
        prediction = self.apply_log_tke_ratio(
            base_prediction, log_ratio, float(self.config.correction_scale)
        )
        half_width = half_width * float(self.config.interval_width_scale)
        if return_interval:
            return prediction, half_width
        return prediction


class TemporalFrequencyTransferCorrector(nn.Module):
    """Infer complex temporal-frequency transfer maps from measured history.

    Unlike a scalar TKE retraction, this operator can move energy between the
    base trajectory's phase quadratures independently at every spatial point,
    velocity component, and non-DC temporal frequency. The frozen base still
    supplies a conservative trajectory and uncertainty interval; a zero head
    is therefore exactly the retained model.
    """

    feature_channels = 56

    def __init__(self, config: TemporalTransferConfig) -> None:
        super().__init__()
        self.config = config
        self.base = MaskedSpectralPosteriorRefiner(
            SpectralPosteriorConfig.from_dict(config.base)
        )
        self.base.requires_grad_(False)
        self.base.eval()
        self.lift = nn.Conv2d(self.feature_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        output_channels = 2 * config.frequency_bins * 2
        self.transfer_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, output_channels, 1),
        )
        nn.init.zeros_(self.transfer_head[-1].weight)
        nn.init.zeros_(self.transfer_head[-1].bias)
        y = torch.linspace(-1.0, 1.0, 32)
        x = torch.linspace(-1.0, 1.0, 64)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer(
            "spatial_coordinates", torch.stack((grid_y, grid_x), dim=0), persistent=True
        )

    def train(self, mode: bool = True) -> TemporalFrequencyTransferCorrector:
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def temporal_tke(velocity: torch.Tensor) -> torch.Tensor:
        return TKEMapCorrector.temporal_tke(velocity)

    def features(self, flow: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        uv = flow[..., :2]
        base_uv = base_prediction[..., :2]
        sequence = uv.permute(0, 1, 4, 2, 3).reshape(flow.shape[0], 40, 32, 64)
        observed_mean = uv.mean(dim=1)
        observed_std = uv.std(dim=1)
        observed_trend = uv[:, -1] - uv[:, 0]
        base_mean = base_uv.mean(dim=1)
        base_std = base_uv.std(dim=1)
        spatial = torch.cat(
            (observed_mean, observed_std, observed_trend, base_mean, base_std), dim=-1
        ).permute(0, 3, 1, 2)
        energy = torch.stack(
            (
                self.temporal_tke(uv),
                self.temporal_tke(uv[:, :10]),
                self.temporal_tke(uv[:, 10:]),
                self.temporal_tke(base_uv),
            ),
            dim=1,
        )
        energy = torch.log1p(energy.clamp_min(0.0) / 1e-3)
        coordinates = self.spatial_coordinates.to(flow.dtype).unsqueeze(0)
        coordinates = coordinates.expand(flow.shape[0], -1, -1, -1)
        result = torch.cat((sequence, spatial, energy, coordinates), dim=1)
        if result.shape[1] != self.feature_channels:
            raise RuntimeError(f"unexpected transfer feature channels: {result.shape[1]}")
        return result

    def predict_transfer(
        self, flow: torch.Tensor, base_prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.lift(self.features(flow, base_prediction))
        for block in self.blocks:
            hidden = block(hidden)
        batch, _, height, width = hidden.shape
        raw = self.transfer_head(self.final_norm(hidden)).reshape(
            batch, 2, self.config.frequency_bins, 2, height, width
        )
        log_amplitude = float(self.config.maximum_log_amplitude) * torch.tanh(
            raw[:, :, :, 0] / max(float(self.config.maximum_log_amplitude), 1e-6)
        )
        phase = float(self.config.maximum_phase_shift) * torch.tanh(
            raw[:, :, :, 1] / max(float(self.config.maximum_phase_shift), 1e-6)
        )
        return log_amplitude, phase

    @staticmethod
    def apply_transfer(
        prediction: torch.Tensor,
        log_amplitude: torch.Tensor,
        phase: torch.Tensor,
        scale: float,
        phase_scale: float | None = None,
    ) -> torch.Tensor:
        uv = prediction[..., :2].float()
        temporal_mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - temporal_mean
        spectrum = torch.fft.rfft(fluctuation, dim=1, norm="ortho")
        bins = min(log_amplitude.shape[2], spectrum.shape[1] - 1)
        scaled_amplitude = float(scale) * log_amplitude[:, :, :bins]
        resolved_phase_scale = float(scale) if phase_scale is None else float(phase_scale)
        scaled_phase = resolved_phase_scale * phase[:, :, :bins]
        multiplier = torch.polar(torch.exp(scaled_amplitude), scaled_phase)
        multiplier = multiplier.permute(0, 2, 3, 4, 1)
        corrected_spectrum = torch.cat(
            (
                spectrum[:, :1],
                spectrum[:, 1 : bins + 1] * multiplier,
                spectrum[:, bins + 1 :],
            ),
            dim=1,
        )
        corrected = temporal_mean + torch.fft.irfft(
            corrected_spectrum, n=uv.shape[1], dim=1, norm="ortho"
        )
        return torch.cat((corrected, torch.zeros_like(corrected[..., :1])), dim=-1)

    def components(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base_prediction, half_width = self.base(flow, return_interval=True)
        log_amplitude, phase = self.predict_transfer(flow, base_prediction)
        return base_prediction, half_width, log_amplitude, phase

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base_prediction, half_width, log_amplitude, phase = self.components(flow)
        prediction = self.apply_transfer(
            base_prediction,
            log_amplitude,
            phase,
            float(self.config.correction_scale),
            self.config.phase_scale,
        )
        half_width = half_width * float(self.config.interval_width_scale)
        if return_interval:
            return prediction, half_width
        return prediction


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
