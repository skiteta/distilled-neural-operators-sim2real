from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PhysicsPreservedTransportConfig:
    """Small training-only joint map for 20-frame flow pairs.

    The map is residual and exactly identity at initialization.  Its two field
    branches share a descriptor of the complete (input, future) pair, so this
    is a joint product-space transport rather than two unrelated marginal
    normalizers.  It is deliberately absent from the deployed forecaster.
    """

    steps: int = 1000
    batch_size: int = 4
    width: int = 32
    depth: int = 3
    maximum_residual_std: float = 0.50
    map_learning_rate: float = 2.0e-4
    potential_learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 2.0e-6
    transport_weight: float = 0.02
    dual_weight: float = 0.20
    observable_weight: float = 0.10
    physics_weight: float = 0.20
    potential_clip: float = 0.05

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhysicsPreservedTransportConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair_descriptor(observed: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    pieces = []
    for field in (observed, future):
        mean = field.mean(dim=(1, 2, 3))
        std = field.std(dim=(1, 2, 3)).clamp_min(1e-5)
        pieces.extend((torch.asinh(mean), torch.log(std)))
    return torch.cat(pieces, dim=-1)


def observable_summary(observed: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Fixed-gauge, differentiable velocity summaries for distribution checks."""

    pieces = []
    for field in (observed[..., :2], future[..., :2]):
        mean = field.mean(dim=(1, 2, 3))
        std = field.std(dim=(1, 2, 3))
        temporal = (field[:, 1:] - field[:, :-1]).square().mean(
            dim=(1, 2, 3)
        ).clamp_min(1e-12).sqrt()
        grad_y = (field[:, :, 1:] - field[:, :, :-1]).square().mean(
            dim=(1, 2, 3)
        ).clamp_min(1e-12).sqrt()
        grad_x = (field[:, :, :, 1:] - field[:, :, :, :-1]).square().mean(
            dim=(1, 2, 3)
        ).clamp_min(1e-12).sqrt()
        pieces.extend((mean, std, temporal, grad_y, grad_x))
    return torch.cat(pieces, dim=-1)


class _ResidualFieldMap(nn.Module):
    def __init__(self, width: int, depth: int, maximum_residual_std: float) -> None:
        super().__init__()
        self.maximum_residual_std = float(maximum_residual_std)
        self.lift = nn.Conv2d(60, width, 1)
        self.film = nn.Sequential(
            nn.Linear(12, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8 if width % 8 == 0 else 1, width),
                    nn.Conv2d(width, width, 5, padding=2, groups=width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 1),
                )
                for _ in range(depth)
            ]
        )
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.full((1, width, 1, 1), 1.0e-2)) for _ in range(depth)]
        )
        self.project = nn.Conv2d(width, 60, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, field: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width, channels = field.shape
        if (steps, channels) != (20, 3):
            raise ValueError(f"expected [B,20,H,W,3], got {tuple(field.shape)}")
        mean = field.mean(dim=(1, 2, 3), keepdim=True)
        std = field.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (field - mean) / std
        hidden = normalized.permute(0, 1, 4, 2, 3).reshape(batch, 60, height, width)
        hidden = self.lift(hidden)
        scale, shift = self.film(descriptor).chunk(2, dim=-1)
        hidden = hidden * (1.0 + 0.10 * torch.tanh(scale)[:, :, None, None])
        hidden = hidden + 0.10 * torch.tanh(shift)[:, :, None, None]
        for block, residual_scale in zip(self.blocks, self.scales):
            hidden = hidden + residual_scale * block(hidden)
        delta = self.project(hidden).reshape(batch, steps, channels, height, width)
        delta = delta.permute(0, 1, 3, 4, 2)
        return field + self.maximum_residual_std * std * torch.tanh(delta)


class PairTransport(nn.Module):
    """Joint input/output transport T=(T_k,T_u)."""

    def __init__(self, config: PhysicsPreservedTransportConfig) -> None:
        super().__init__()
        self.config = config
        self.input_map = _ResidualFieldMap(
            config.width, config.depth, config.maximum_residual_std
        )
        self.future_map = _ResidualFieldMap(
            config.width, config.depth, config.maximum_residual_std
        )

    def forward(
        self, observed: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = _pair_descriptor(observed, future)
        return (
            self.input_map(observed, descriptor),
            self.future_map(future, descriptor),
        )


class PairPotential(nn.Module):
    """Kantorovich-style scalar potential on a complete flow pair."""

    def __init__(self, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.lift = nn.Conv2d(120, width, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8 if width % 8 == 0 else 1, width),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                )
                for _ in range(depth)
            ]
        )
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.full((1, width, 1, 1), 1.0e-2)) for _ in range(depth)]
        )
        self.descriptor = nn.Sequential(
            nn.Linear(12, width), nn.GELU(), nn.Linear(width, width)
        )
        self.head = nn.Sequential(nn.GELU(), nn.Linear(width, 1))

    def forward(self, observed: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width, channels = observed.shape
        pair = torch.cat((observed, future), dim=1)
        mean = pair.mean(dim=(1, 2, 3), keepdim=True)
        std = pair.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (pair - mean) / std
        hidden = normalized.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * 2 * channels, height, width
        )
        hidden = F.avg_pool2d(hidden, kernel_size=(4, 4), stride=(4, 4))
        hidden = self.lift(hidden)
        for block, residual_scale in zip(self.blocks, self.scales):
            hidden = hidden + residual_scale * block(hidden)
        pooled = hidden.mean(dim=(2, 3)) + self.descriptor(
            _pair_descriptor(observed, future)
        )
        return self.head(pooled).squeeze(-1)


def normalized_transport_cost(
    mapped_observed: torch.Tensor,
    mapped_future: torch.Tensor,
    source_observed: torch.Tensor,
    source_future: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for mapped, source in (
        (mapped_observed, source_observed),
        (mapped_future, source_future),
    ):
        scale = source.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        terms.append(((mapped - source) / scale).square().mean())
    return torch.stack(terms).mean()


def physics_consistency(
    source_operator: nn.Module,
    mapped_observed: torch.Tensor,
    mapped_future: torch.Tensor,
) -> torch.Tensor:
    predicted = source_operator(mapped_observed)[..., :2]
    target = mapped_future[..., :2]
    numerator = (predicted - target).square().mean(dim=(1, 2, 3, 4))
    denominator = target.square().mean(dim=(1, 2, 3, 4)).clamp_min(1e-8)
    return (numerator / denominator).mean()


def standardized_observable_loss(
    mapped_observed: torch.Tensor,
    mapped_future: torch.Tensor,
    target_observed: torch.Tensor,
    target_future: torch.Tensor,
) -> torch.Tensor:
    mapped = observable_summary(mapped_observed, mapped_future)
    target = observable_summary(target_observed, target_future).detach()
    scale = target.std(dim=0, unbiased=False).clamp_min(1e-3)
    mean_loss = (((mapped.mean(0) - target.mean(0)) / scale) ** 2).mean()
    mapped_centered = (mapped - mapped.mean(0)) / scale
    target_centered = (target - target.mean(0)) / scale
    mapped_cov = mapped_centered.T @ mapped_centered / max(1, len(mapped) - 1)
    target_cov = target_centered.T @ target_centered / max(1, len(target) - 1)
    covariance_loss = (mapped_cov - target_cov).square().mean()
    return mean_loss + 0.05 * covariance_loss
