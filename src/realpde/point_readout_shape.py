"""Shape residual using the frozen velocity head's existing features."""
import torch
from torch import nn
from .baseline_shape_correction import preserve_moments

class PointReadoutShape(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.readout = nn.Conv2d(channels, 40, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features, baseline):
        z = self.readout(features).reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        f = baseline - baseline.mean(1, keepdim=True)
        rms = f.square().sum(-1, keepdim=True).mean(1, keepdim=True).sqrt()
        return preserve_moments(baseline + .3 * z.tanh() * rms, baseline)

def point_with_readout_features(model, observed):
    captured = []
    handle = model.velocity_head[1].register_forward_hook(lambda m, i, o: captured.append(o))
    try:
        point = model(observed).float()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('expected one velocity-head feature field')
    return point, captured[0]
