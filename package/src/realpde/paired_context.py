from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import OperatorBlock


def context_descriptor(flow: torch.Tensor, pool_size: tuple[int, int] = (4, 8)) -> torch.Tensor:
    """Observed-only descriptor used to identify the operating condition.

    The descriptor deliberately uses the complete measured field instead of an
    upstream strip: the PIV files contain zero-valued unobserved pixels near the
    domain boundary.  No future frame or file-name metadata enters this path.
    """

    if flow.ndim != 5 or flow.shape[-1] < 2:
        raise ValueError("flow must have shape [B,T,H,W,C>=2]")
    uv = flow[..., :2]
    temporal_mean = uv.mean(dim=1)
    fluctuation = uv - temporal_mean[:, None]
    temporal_std = fluctuation.square().mean(dim=1).clamp_min(1e-10).sqrt()
    local_tke = 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1, keepdim=False)
    edge = min(5, uv.shape[1] // 2)
    trend = uv[:, -edge:].mean(dim=1) - uv[:, :edge].mean(dim=1)
    maps = torch.cat(
        (temporal_mean, temporal_std, trend, local_tke[..., None]), dim=-1
    ).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(maps, pool_size)
    return pooled.flatten(1)


@dataclass(frozen=True)
class ContextRidgeConfig:
    pool_y: int = 4
    pool_x: int = 8
    feature_count: int = 224
    penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservedConditionRidge(nn.Module):
    """Fixed ridge map from observed PIV history to (Re, AoA).

    Outputs physical units.  All tensors required at inference are buffers so a
    packaged model cannot accidentally depend on training files or metadata.
    """

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        coefficients: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
        *,
        pool_size: tuple[int, int] = (4, 8),
        minimum: tuple[float, float] = (2968.0, 0.0),
        maximum: tuple[float, float] = (27975.0, 20.0),
    ) -> None:
        super().__init__()
        self.pool_size = tuple(int(value) for value in pool_size)
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_scale", feature_scale.float().clamp_min(1e-6))
        self.register_buffer("coefficients", coefficients.float())
        self.register_buffer("target_mean", target_mean.float())
        self.register_buffer("target_scale", target_scale.float().clamp_min(1e-6))
        self.register_buffer("minimum", torch.tensor(minimum, dtype=torch.float32))
        self.register_buffer("maximum", torch.tensor(maximum, dtype=torch.float32))

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        features = context_descriptor(flow, self.pool_size)
        standardized = (features - self.feature_mean) / self.feature_scale
        normalized = standardized @ self.coefficients
        physical = normalized * self.target_scale + self.target_mean
        return torch.maximum(torch.minimum(physical, self.maximum), self.minimum)


class SimulationPrototypeBank(nn.Module):
    """Differentiable lookup of CFD regime prototypes from inferred context."""

    def __init__(
        self,
        prototypes: torch.Tensor,
        regimes: torch.Tensor,
        *,
        re_spacing: float = 1275.0,
        aoa_spacing: float = 5.0,
        temperature: float = 0.20,
    ) -> None:
        super().__init__()
        if prototypes.ndim != 4:
            raise ValueError("prototypes must have shape [R,C,H,W]")
        if regimes.shape != (prototypes.shape[0], 2):
            raise ValueError("regimes must have shape [R,2]")
        self.register_buffer("prototypes", prototypes.float())
        self.register_buffer("regimes", regimes.float())
        self.register_buffer(
            "spacing", torch.tensor((re_spacing, aoa_spacing), dtype=torch.float32)
        )
        self.temperature = float(temperature)

    @property
    def channels(self) -> int:
        return int(self.prototypes.shape[1])

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance = ((condition[:, None] - self.regimes[None]) / self.spacing).square().sum(-1)
        weights = torch.softmax(-0.5 * distance / self.temperature, dim=1)
        prototype = torch.einsum("br,rchw->bchw", weights, self.prototypes)
        return prototype, weights


@dataclass(frozen=True)
class PairedContextAdapterConfig:
    in_steps: int = 20
    out_steps: int = 20
    width: int = 48
    depth: int = 4
    spectral_rank: int = 8
    modes_y: int = 10
    modes_x: int = 18
    expansion: int = 2
    dropout: float = 0.02
    maximum_correction_std: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PairedContextResidualAdapter(nn.Module):
    """Small CFD-conditioned residual head around a frozen real forecaster.

    CFD enters as a regime-specific spatial/spectral prototype.  Inferred
    (Re, AoA) also FiLM-modulates every operator block.  The final layer is
    zero-initialized, making the initial model exactly the frozen anchor.
    """

    def __init__(self, config: PairedContextAdapterConfig, prototype_channels: int) -> None:
        super().__init__()
        self.config = config
        self.prototype_channels = int(prototype_channels)
        observed_channels = config.in_steps * 2
        forecast_channels = config.out_steps * 2
        physics_channels = 9
        coordinate_channels = 2
        in_channels = (
            observed_channels
            + forecast_channels
            + physics_channels
            + coordinate_channels
            + self.prototype_channels
        )
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
        self.film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, config.width),
                    nn.GELU(),
                    nn.Linear(config.width, 2 * config.width),
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)
        self.head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, config.out_steps * 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _coordinates(flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width, _ = flow.shape
        yy = torch.linspace(-1, 1, height, device=flow.device, dtype=flow.dtype)
        xx = torch.linspace(-1, 1, width, device=flow.device, dtype=flow.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0)[None].expand(batch, -1, -1, -1)

    def _features(
        self,
        observed: torch.Tensor,
        anchor: torch.Tensor,
        prototype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uv = observed[..., :2]
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_observed = (uv - mean) / scale
        normalized_anchor = (anchor[..., :2] - mean) / scale
        batch, _, height, width, _ = observed.shape
        observed_flat = normalized_observed.permute(0, 1, 4, 2, 3).reshape(
            batch, -1, height, width
        )
        anchor_flat = normalized_anchor.permute(0, 1, 4, 2, 3).reshape(
            batch, -1, height, width
        )
        temporal_mean = normalized_observed.mean(dim=1)
        fluctuation = normalized_observed - temporal_mean[:, None]
        variance = fluctuation.square().mean(dim=1)
        temporal_std = variance.clamp_min(1e-8).sqrt()
        local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
        recent = max(0, self.config.in_steps - 5)
        trend = normalized_observed[:, -1] - normalized_observed[:, recent]
        final_fluctuation = normalized_observed[:, -1] - temporal_mean
        physics = torch.cat(
            (temporal_mean, temporal_std, local_tke, trend, final_fluctuation), dim=-1
        ).permute(0, 3, 1, 2)
        features = torch.cat(
            (
                observed_flat,
                anchor_flat,
                physics,
                self._coordinates(observed),
                prototype.to(dtype=observed.dtype),
            ),
            dim=1,
        )
        valid = uv.abs().sum(dim=(1, 4), keepdim=False) > 0.0
        return features, mean, scale, valid

    def forward(
        self,
        observed: torch.Tensor,
        anchor: torch.Tensor,
        prototype: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, _mean, scale, valid = self._features(observed, anchor, prototype)
        hidden = self.lift(features)
        normalized_condition = torch.stack(
            ((condition[:, 0] - 15471.5) / 12503.5, (condition[:, 1] - 10.0) / 10.0),
            dim=1,
        ).to(dtype=hidden.dtype)
        for block, film in zip(self.blocks, self.film):
            parameters = film(normalized_condition)
            gamma, beta = parameters.chunk(2, dim=1)
            hidden = hidden * (1.0 + 0.10 * torch.tanh(gamma[:, :, None, None]))
            hidden = hidden + 0.10 * beta[:, :, None, None]
            hidden = block(hidden)
        raw = self.head(self.final_norm(hidden)).reshape(
            observed.shape[0], self.config.out_steps, 2, observed.shape[2], observed.shape[3]
        ).permute(0, 1, 3, 4, 2)
        correction = (
            torch.tanh(raw)
            * scale
            * float(self.config.maximum_correction_std)
            * valid[:, None, :, :, None]
        )
        velocity = anchor[..., :2] + correction
        prediction = torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)
        return prediction, correction

