"""Function-preserving additive capacity for a FastInterval radius head.

Two independent hidden streams sum before the usual shift/radius nonlinearities.
The second stream starts with a zero final projection and has no final bias.
The original stream remains trainable; inference uses released-window features.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .fast_interval import FastIntervalConfig, FastIntervalHead


class AdditiveHeadNetwork(nn.Module):
    def __init__(self, base, extra):
        super().__init__()
        self.base = base
        self.extra = extra

    def forward(self, diagnostics):
        return self.base(diagnostics) + self.extra(diagnostics)


class ExpandedFastIntervalHead(FastIntervalHead):
    architecture = "additive_capacity_2x"

    def __init__(self, config: FastIntervalConfig):
        super().__init__(config)
        extra = FastIntervalHead(config).network
        extra[-1] = nn.Conv2d(config.hidden_channels, config.out_steps * 4, 1, bias=False)
        self.network = AdditiveHeadNetwork(self.network, extra)

    @classmethod
    def expand(cls, original: FastIntervalHead, seed: int):
        # Initialize on CPU and restore CPU RNG; do not change CUDA RNG/settings.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = cls(copy.deepcopy(original.config))
        head.network.base.load_state_dict(original.network.state_dict(), strict=True)
        with torch.no_grad():
            head.network.extra[-1].weight.zero_()
        return head


def load_expanded_head(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ExpandedFastIntervalHead.architecture:
        raise ValueError("checkpoint is not an additive-capacity head")
    model = ExpandedFastIntervalHead(FastIntervalConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)
