"""Warm-started nonlinear residual on already-computed point features."""
import torch
from torch import nn
from .point_readout_shape import PointReadoutShape
from .baseline_shape_correction import preserve_moments

class PointReadoutRefinement(PointReadoutShape):
    def __init__(self, channels, width=16):
        super().__init__(channels)
        self.refine = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU(), nn.Conv2d(width, 40, 1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, features, baseline):
        z = self.readout(features) + self.refine(features)
        z = z.reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        f = baseline - baseline.mean(1, keepdim=True)
        rms = f.square().sum(-1, keepdim=True).mean(1, keepdim=True).sqrt()
        return preserve_moments(baseline + .3 * z.tanh() * rms, baseline)
