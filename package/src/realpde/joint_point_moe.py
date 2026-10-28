from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import HybridFlowOperator, ModelConfig


@dataclass(frozen=True)
class JointPointMoEConfig:
    """Frozen C346 query-local, jointly specialized point mixture."""

    base: dict[str, Any]
    experts: int = 3
    expert_hidden: int = 64
    residual_cap: float = 0.25
    gate_hidden: int = 128
    gate_pool_y: int = 4
    gate_pool_x: int = 8
    gate_temperature: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> JointPointMoEConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JointPointMixture(nn.Module):
    """Hybrid operator with co-adapted residual decoders and a local gate.

    Every route depends only on the window being answered.  The gate consumes
    a pooled copy of that window's full shared latent map; no metadata, batch
    statistic, cross-window state, or target is available at inference.
    """

    def __init__(self, config: JointPointMoEConfig) -> None:
        super().__init__()
        if config.experts != 3:
            raise ValueError("C346 is frozen to exactly three experts")
        self.config = config
        base_config = ModelConfig.from_dict(config.base)
        self.base = HybridFlowOperator(base_config)
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base_config.width, config.expert_hidden, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(config.expert_hidden, base_config.out_steps * 2, 1),
                )
                for _ in range(config.experts)
            ]
        )
        gate_features = base_config.width * config.gate_pool_y * config.gate_pool_x
        self.gate_pool = nn.AdaptiveAvgPool2d((config.gate_pool_y, config.gate_pool_x))
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_features, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, config.experts),
        )
        # A single seeded construction yields different but near-identity
        # residual branches.  Zero gate logits begin with equal route mass.
        for head in self.expert_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict, strict=True)

    def _shared(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self.base._features(flow)
        hidden = self.base.lift(features)
        for index, block in enumerate(self.base.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.base.blocks) // 2):
                hidden = self.base.multiscale(hidden)
        hidden = self.base.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.base.velocity_head(hidden).reshape(
            batch, self.base.config.out_steps, 2, height, width
        )
        correction = correction.permute(0, 1, 3, 4, 2)
        base_normalized = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent_trend = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4 * uv_std)
        trend = recent_trend * self.base.trend_scale[None, :, None, None, :]
        common_uv = (base_normalized + trend + correction) * uv_std + uv_mean
        return hidden, common_uv, uv_mean, uv_std

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        hidden, common_uv, _uv_mean, uv_std = self._shared(flow)
        batch, _, height, width = hidden.shape
        residuals: list[torch.Tensor] = []
        for head in self.expert_heads:
            raw = head(hidden).reshape(batch, self.base.config.out_steps, 2, height, width)
            raw = raw.permute(0, 1, 3, 4, 2)
            residuals.append(self.config.residual_cap * torch.tanh(raw) * uv_std)
        expert_uv = common_uv[:, None] + torch.stack(residuals, dim=1)
        pressure = torch.zeros_like(expert_uv[..., :1])
        experts = torch.cat((expert_uv, pressure), dim=-1)

        logits = self.gate(self.gate_pool(hidden))
        weights = F.softmax(logits / self.config.gate_temperature, dim=1)
        prediction = (experts * weights[:, :, None, None, None, None]).sum(dim=1)

        raw_width = self.base.width_head(hidden).reshape(
            batch, self.base.config.out_steps, 2, height, width
        )
        raw_width = raw_width.permute(0, 1, 3, 4, 2)
        half_width_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.base.interval_temperature.clamp_min(1e-3)
        )
        half_width = torch.cat(
            (half_width_uv, torch.zeros_like(half_width_uv[..., :1])), dim=-1
        )
        if return_details:
            return {
                "prediction": prediction,
                "half_width": half_width,
                "experts": experts,
                "logits": logits,
                "weights": weights,
                "common": torch.cat(
                    (common_uv, torch.zeros_like(common_uv[..., :1])), dim=-1
                ),
            }
        if return_interval:
            return prediction, half_width
        return prediction

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self(input_array)
