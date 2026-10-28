"""Combine two existing input projections; no new fitted parameters."""
import torch
from torch import nn
from .baseline_shape_correction import preserve_moments

class FusedPointReadout(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.joined = nn.Conv2d(channels, 56, 1)
        self.up = nn.Conv2d(16, 40, 1)

    def forward(self, features, baseline):
        q = self.joined(features)
        z = q[:, :40] + self.up(torch.nn.functional.gelu(q[:, 40:]))
        z = z.reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        f = baseline - baseline.mean(1, keepdim=True)
        rms = f.square().sum(-1, keepdim=True).mean(1, keepdim=True).sqrt()
        return preserve_moments(baseline + .3 * z.tanh() * rms, baseline)
