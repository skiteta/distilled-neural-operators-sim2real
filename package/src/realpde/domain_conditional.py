from __future__ import annotations

import copy

import torch
from torch import nn

from .model import HybridFlowOperator, ModelConfig


class DomainConditionalHybrid(HybridFlowOperator):
    """Hybrid operator with domain-specific lifts and shared real dynamics."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.sim_lift = copy.deepcopy(self.lift)
        self.context_film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, config.width),
                    nn.GELU(),
                    nn.Linear(config.width, 2 * config.width),
                )
                for _ in range(config.depth)
            ]
        )
        for film in self.context_film:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    def initialize_from_anchors(
        self,
        real_anchor: HybridFlowOperator,
        sim_anchor: HybridFlowOperator,
    ) -> None:
        missing, unexpected = self.load_state_dict(real_anchor.state_dict(), strict=False)
        expected_missing = {
            "sim_lift.weight",
            "sim_lift.bias",
            *{
                f"context_film.{index}.{layer}.{name}"
                for index in range(len(self.context_film))
                for layer in (0, 2)
                for name in ("weight", "bias")
            },
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"unexpected anchor initialization keys: missing={missing}, unexpected={unexpected}"
            )
        self.sim_lift.load_state_dict(sim_anchor.lift.state_dict(), strict=True)

    @staticmethod
    def _normalized_condition(condition: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            (
                (condition[:, 0] - 15471.5) / 12503.5,
                (condition[:, 1] - 10.0) / 10.0,
            ),
            dim=1,
        ).to(dtype=dtype)

    def forward(
        self,
        flow: torch.Tensor,
        condition: torch.Tensor,
        *,
        domain: str = "real",
    ) -> torch.Tensor:
        if domain not in {"real", "sim"}:
            raise ValueError("domain must be 'real' or 'sim'")
        features, uv_mean, uv_std = self._features(flow)
        hidden = (self.lift if domain == "real" else self.sim_lift)(features)
        normalized = self._normalized_condition(condition, hidden.dtype)
        for index, (block, film) in enumerate(zip(self.blocks, self.context_film)):
            gamma, beta = film(normalized).chunk(2, dim=1)
            hidden = hidden * (1.0 + 0.05 * torch.tanh(gamma[:, :, None, None]))
            hidden = hidden + 0.05 * beta[:, :, None, None]
            hidden = block(hidden)
            if index + 1 == max(1, len(self.blocks) // 2):
                hidden = self.multiscale(hidden)
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4 * uv_std)
        trend = recent * self.trend_scale[None, :, None, None, :]
        velocity = (base + trend + correction) * uv_std + uv_mean
        return torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)
