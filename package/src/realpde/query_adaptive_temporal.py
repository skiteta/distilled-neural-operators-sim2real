from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import HybridFlowOperator, ModelConfig


@dataclass(frozen=True)
class QueryAdaptiveTemporalConfig:
    """Configuration for causal query-conditioned Hybrid modulation."""

    base: dict[str, Any]
    context_width: int = 48
    context_depth: int = 3
    film_hidden: int = 64
    modulation_cap: float = 0.10
    local_cap: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalContextBlock(nn.Module):
    """Separable temporal/local transport on an explicit observed-time axis."""

    def __init__(self, width: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.temporal = nn.Conv3d(
            width, width, (5, 1, 1), padding=(2, 0, 0), groups=width
        )
        self.spatial = nn.Conv3d(
            width, width, (1, 5, 5), padding=(0, 2, 2), groups=width
        )
        self.channel = nn.Conv3d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, 4 * width, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Conv3d(2 * width, width, 1),
        )
        self.operator_scale = nn.Parameter(
            torch.full((1, width, 1, 1, 1), 1e-2)
        )
        self.ffn_scale = nn.Parameter(
            torch.full((1, width, 1, 1, 1), 1e-2)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        update = (
            self.temporal(normalized)
            + self.spatial(normalized)
            + self.channel(normalized)
        )
        value = value + self.operator_scale * update
        return value + self.ffn_scale * self.ffn(self.norm2(value))


class QueryAdaptiveTemporalHybrid(HybridFlowOperator):
    """Full Hybrid with zero-initialized query-specific temporal modulation."""

    def __init__(self, config: QueryAdaptiveTemporalConfig) -> None:
        self.adaptive_config = config
        base_config = ModelConfig.from_dict(config.base)
        super().__init__(base_config)
        width = int(config.context_width)
        groups = 8 if width % 8 == 0 else 1
        self.temporal_lift = nn.Sequential(
            nn.Conv3d(
                5,
                width,
                kernel_size=(3, 5, 5),
                stride=(1, 4, 4),
                padding=(1, 2, 2),
            ),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalContextBlock(width) for _ in range(int(config.context_depth))
        )
        self.temporal_attention = nn.Conv3d(width, 1, 1)
        self.temporal_global_norm = nn.LayerNorm(width)
        self.local_input = nn.Conv2d(width, base_config.width, 1)
        self.local_midpoint = nn.Conv2d(width, base_config.width, 1)
        self.film = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, int(config.film_hidden)),
                nn.GELU(),
                nn.Linear(int(config.film_hidden), 2 * base_config.width),
            )
            for _ in range(base_config.depth)
        )
        nn.init.zeros_(self.local_input.weight)
        nn.init.zeros_(self.local_input.bias)
        nn.init.zeros_(self.local_midpoint.weight)
        nn.init.zeros_(self.local_midpoint.bias)
        for generator in self.film:
            nn.init.zeros_(generator[-1].weight)
            nn.init.zeros_(generator[-1].bias)

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unexpected = sorted(set(state_dict) - set(current))
        if unexpected:
            raise RuntimeError(f"unexpected base state keys: {unexpected}")
        for name, value in state_dict.items():
            if current[name].shape != value.shape:
                raise RuntimeError(f"base shape mismatch for {name}")
            current[name] = value
        self.load_state_dict(current, strict=True)

    @staticmethod
    def _temporal_inputs(flow: torch.Tensor) -> torch.Tensor:
        uv = flow[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (uv - mean) / scale
        difference = torch.zeros_like(normalized)
        difference[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        batch, steps, height, width, _channels = normalized.shape
        time = torch.linspace(
            -1.0, 0.0, steps, device=flow.device, dtype=normalized.dtype
        )[None, :, None, None, None]
        time = time.expand(batch, steps, height, width, 1)
        return torch.cat((normalized, difference, time), dim=-1).permute(
            0, 4, 1, 2, 3
        )

    def _temporal_context(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.temporal_lift(self._temporal_inputs(flow))
        for block in self.temporal_blocks:
            context = block(context)
        logits = self.temporal_attention(context)
        weights = torch.softmax(logits.float(), dim=2).to(context.dtype)
        local = (context * weights).sum(dim=2)
        global_context = self.temporal_global_norm(
            context.mean(dim=(2, 3, 4))
        )
        return local, global_context, weights

    def _local_update(
        self,
        local: torch.Tensor,
        projection: nn.Conv2d,
        size: tuple[int, int],
    ) -> torch.Tensor:
        value = projection(local)
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
        return float(self.adaptive_config.local_cap) * torch.tanh(value)

    def _film_parameters(
        self, global_context: torch.Tensor, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.film[index](global_context).chunk(2, dim=1)
        cap = float(self.adaptive_config.modulation_cap)
        return cap * torch.tanh(gamma), cap * torch.tanh(beta)

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raw = flow
        features, uv_mean, uv_std = self._features(raw)
        local, global_context, _weights = self._temporal_context(raw)
        hidden = self.lift(features)
        hidden = hidden + self._local_update(
            local, self.local_input, hidden.shape[-2:]
        )
        midpoint = max(1, len(self.blocks) // 2)
        for index, block in enumerate(self.blocks):
            gamma, beta = self._film_parameters(global_context, index)
            hidden = hidden * (1.0 + gamma[:, :, None, None])
            hidden = hidden + beta[:, :, None, None]
            hidden = block(hidden)
            if index + 1 == midpoint:
                hidden = self.multiscale(hidden)
                hidden = hidden + self._local_update(
                    local, self.local_midpoint, hidden.shape[-2:]
                )
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (raw[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (raw[:, -1:, ..., :2] - raw[:, -5:-4, ..., :2]) / (
            4.0 * uv_std
        )
        trend = recent * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base + trend + correction) * uv_std + uv_mean
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )
        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        return prediction, half

    @torch.inference_mode()
    def adaptation_diagnostics(self, flow: torch.Tensor) -> dict[str, float]:
        local, global_context, weights = self._temporal_context(flow)
        local_input = self._local_update(
            local, self.local_input, (flow.shape[2], flow.shape[3])
        )
        local_mid = self._local_update(
            local, self.local_midpoint, (flow.shape[2], flow.shape[3])
        )
        films = [
            torch.cat(self._film_parameters(global_context, index), dim=1)
            for index in range(len(self.film))
        ]
        film = torch.stack(films, dim=1)
        entropy = -(
            weights.float() * weights.float().clamp_min(1e-12).log()
        ).sum(dim=2)
        return {
            "context_std": float(global_context.float().std(unbiased=False)),
            "local_input_rms": float(local_input.float().square().mean().sqrt()),
            "local_midpoint_rms": float(local_mid.float().square().mean().sqrt()),
            "film_rms": float(film.float().square().mean().sqrt()),
            "film_max_abs": float(film.float().abs().max()),
            "temporal_attention_entropy": float(entropy.mean()),
            "temporal_attention_peak": float(
                weights.float().max(dim=2).values.mean()
            ),
        }

    def predict(
        self,
        input_array: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
