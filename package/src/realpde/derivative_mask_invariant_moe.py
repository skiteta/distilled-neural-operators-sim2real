from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import HybridFlowOperator, LowRankSpectralMix2d, ModelConfig


@dataclass(frozen=True)
class DerivativeMaskInvariantConfig:
    base: dict[str, Any]
    experts: int = 3
    derivative_channels: int = 10
    expert_width: int = 64
    spectral_rank: int = 12
    gate_hidden: int = 96
    gate_temperature: float = 0.7
    residual_cap: float = 0.20

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DerivativeMaskInvariantConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def physical_derivatives(normalized_last: torch.Tensor) -> torch.Tensor:
    """Ten periodic central-difference channels from normalized [B,H,W,2] u/v."""
    field = normalized_last.permute(0, 3, 1, 2)
    xp = torch.roll(field, shifts=-1, dims=3)
    xm = torch.roll(field, shifts=1, dims=3)
    yp = torch.roll(field, shifts=-1, dims=2)
    ym = torch.roll(field, shifts=1, dims=2)
    dx = 0.5 * (xp - xm)
    dy = 0.5 * (yp - ym)
    dxx = xp - 2.0 * field + xm
    dyy = yp - 2.0 * field + ym
    xpypp = torch.roll(xp, shifts=-1, dims=2)
    xpym = torch.roll(xp, shifts=1, dims=2)
    xmypp = torch.roll(xm, shifts=-1, dims=2)
    xmym = torch.roll(xm, shifts=1, dims=2)
    dxy = 0.25 * (xpypp - xpym - xmypp + xmym)
    return torch.stack(
        (dx[:, 0], dx[:, 1], dy[:, 0], dy[:, 1], dxx[:, 0], dxx[:, 1], dyy[:, 0], dyy[:, 1], dxy[:, 0], dxy[:, 1]),
        dim=1,
    )


class DerivativeMaskedSpectralExpert(nn.Module):
    def __init__(self, hidden_width: int, config: DerivativeMaskInvariantConfig, base: ModelConfig) -> None:
        super().__init__()
        width = config.expert_width
        groups = 8 if width % 8 == 0 else 1
        self.hidden_lift = nn.Conv2d(hidden_width, width, 1)
        self.derivative_lift = nn.Conv2d(config.derivative_channels, width, 1)
        self.norm = nn.GroupNorm(groups, width)
        self.spectral = LowRankSpectralMix2d(width, config.spectral_rank, base.modes_y, base.modes_x)
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, base.out_steps * 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, hidden: torch.Tensor, derivatives: torch.Tensor) -> torch.Tensor:
        state = self.hidden_lift(hidden) + self.derivative_lift(derivatives)
        normalized = self.norm(state)
        state = state + self.spectral(normalized) + self.local(F.gelu(normalized))
        return self.head(F.gelu(self.norm(state)))


class DerivativeMaskInvariantMoE(nn.Module):
    expert_names = ("derivative_first", "derivative_second", "derivative_mixed_cross")
    derivative_names = ("u_x", "v_x", "u_y", "v_y", "u_xx", "v_xx", "u_yy", "v_yy", "u_xy", "v_xy")

    def __init__(self, config: DerivativeMaskInvariantConfig) -> None:
        super().__init__()
        if config.experts != 3 or config.derivative_channels != 10:
            raise ValueError("C406 is frozen to three experts and ten derivatives")
        self.invariant_config = config
        self.config = ModelConfig.from_dict(config.base)
        self.base = HybridFlowOperator(self.config)
        initial = torch.tensor(
            [
                [2, 2, 2, 2, -2, -2, -2, -2, -2, -2],
                [-2, -2, -2, -2, 2, 2, 2, 2, -2, -2],
                [-2, 2, 2, -2, 2, -2, -2, 2, 2, 2],
            ],
            dtype=torch.float32,
        )
        self.mask_logits = nn.Parameter(initial)
        self.experts = nn.ModuleList(
            DerivativeMaskedSpectralExpert(self.config.width, config, self.config)
            for _ in range(config.experts)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.width, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, self.config.out_steps * config.experts),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def masks(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def mask_diversity(self) -> torch.Tensor:
        masks = self.masks()
        normalized = F.normalize(masks, dim=1)
        similarity = normalized @ normalized.T
        off_diagonal = similarity[~torch.eye(len(masks), dtype=torch.bool, device=masks.device)]
        return off_diagonal.mean()

    def mask_sharpness(self) -> torch.Tensor:
        masks = self.masks()
        return (masks * (1.0 - masks)).mean()

    def _shared(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self.base._features(flow)
        hidden = self.base.lift(features)
        for index, block in enumerate(self.base.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.base.blocks) // 2):
                hidden = self.base.multiscale(hidden)
        hidden = self.base.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.base.velocity_head(hidden).reshape(batch, self.config.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        normalized_last = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        normalized_recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4.0 * uv_std)
        trend = normalized_recent * self.base.trend_scale[None, :, None, None, :]
        common = (normalized_last + trend + correction) * uv_std + uv_mean
        derivatives = physical_derivatives(normalized_last[:, 0])
        return hidden, common, uv_mean, uv_std, derivatives

    def forward(self, flow: torch.Tensor, *, return_interval: bool = False, return_details: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        hidden, common, _uv_mean, uv_std, derivatives = self._shared(flow)
        masks = self.masks()
        residual_list = []
        batch = len(flow)
        for index, expert in enumerate(self.experts):
            raw = expert(hidden, derivatives * masks[index][None, :, None, None])
            raw = raw.reshape(batch, self.config.out_steps, 2, hidden.shape[2], hidden.shape[3]).permute(0, 1, 3, 4, 2)
            residual_list.append(float(self.invariant_config.residual_cap) * torch.tanh(raw) * uv_std)
        residuals = torch.stack(residual_list, dim=1)
        expert_uv = common[:, None] + residuals
        logits = self.gate(hidden).reshape(batch, self.config.out_steps, self.invariant_config.experts)
        weights = F.softmax(logits / float(self.invariant_config.gate_temperature), dim=-1)
        prediction_uv = (expert_uv * weights.permute(0, 2, 1)[:, :, :, None, None, None]).sum(dim=1)
        prediction = torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)
        raw_width = self.base.width_head(hidden).reshape(batch, self.config.out_steps, 2, hidden.shape[2], hidden.shape[3]).permute(0, 1, 3, 4, 2)
        half_uv = F.softplus(raw_width) * uv_std * self.base.interval_temperature.clamp_min(1e-3)
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        if return_details:
            return {
                "prediction": prediction,
                "half_width": half,
                "experts": torch.cat((expert_uv, torch.zeros_like(expert_uv[..., :1])), dim=-1),
                "residuals": residuals,
                "logits": logits,
                "weights": weights,
                "masks": masks,
                "derivatives": derivatives,
                "common": torch.cat((common, torch.zeros_like(common[..., :1])), dim=-1),
            }
        if return_interval:
            return prediction, half
        return prediction

    def predict(self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del metadata
        return self(input_array)


__all__ = ["DerivativeMaskInvariantConfig", "DerivativeMaskInvariantMoE", "physical_derivatives"]
