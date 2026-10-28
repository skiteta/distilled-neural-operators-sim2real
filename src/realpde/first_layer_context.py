from __future__ import annotations

import torch
from torch import nn


class FirstLayerContextHybrid(nn.Module):
    """Inject CFD context before a frozen Hybrid operator's shared dynamics."""

    def __init__(self, anchor: nn.Module, prototype_channels: int) -> None:
        super().__init__()
        self.anchor = anchor.requires_grad_(False)
        width = int(anchor.config.width)
        self.prototype_lift = nn.Sequential(
            nn.Conv2d(int(prototype_channels), width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.condition_lift = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, 2 * width)
        )
        self.injection_scale = nn.Parameter(torch.zeros(1, width, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def adapter_parameters(self):
        yield from self.prototype_lift.parameters()
        yield from self.condition_lift.parameters()
        yield self.injection_scale

    def forward(self, flow, prototype, condition):
        features, uv_mean, uv_std = self.anchor._features(flow)
        hidden = self.anchor.lift(features)
        normalized = torch.stack(
            ((condition[:, 0] - 15471.5) / 12503.5, (condition[:, 1] - 10.0) / 10.0),
            dim=1,
        ).to(hidden.dtype)
        gamma, beta = self.condition_lift(normalized).chunk(2, dim=1)
        context = self.prototype_lift(prototype.to(hidden.dtype))
        context = context * (1.0 + 0.10 * torch.tanh(gamma[:, :, None, None]))
        context = context + 0.10 * beta[:, :, None, None]
        injection = torch.tanh(self.injection_scale) * context
        hidden = hidden + injection
        for index, block in enumerate(self.anchor.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.anchor.blocks) // 2):
                hidden = self.anchor.multiscale(hidden)
        hidden = self.anchor.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.anchor.velocity_head(hidden).reshape(
            batch, self.anchor.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4 * uv_std)
        trend = recent * self.anchor.trend_scale[None, :, None, None, :]
        velocity = (base + trend + correction) * uv_std + uv_mean
        prediction = torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)
        return prediction, injection
