"""C665 correction distilled into one expert and one moment projection."""
import numpy as np
import torch
from torch import nn
from .scaled_shape_mixture import FieldExpert, ScaledShapeMixture
from .baseline_shape_correction import preserve_moments


class UnifiedShape(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.experts = nn.ModuleList([FieldExpert(width)])

    inputs = staticmethod(ScaledShapeMixture.inputs)

    def from_features(self, features, rms, baseline):
        z = self.experts[0](features)
        z = z.reshape(len(z), 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        return preserve_moments(baseline + .3 * z.tanh() * rms, baseline)

    def forward(self, observed, baseline):
        features, rms = self.inputs(observed, baseline)
        return self.from_features(features, rms, baseline)

    @torch.inference_mode()
    def predict_numpy(self, observed, baseline, metadata=None):
        if len(observed) != len(baseline):
            raise ValueError('length mismatch')
        if not len(baseline):
            return np.empty_like(baseline)
        device = next(self.parameters()).device
        x = torch.zeros((32, 20, 32, 64, 2), device=device)
        b = torch.zeros_like(x)
        output = []
        for start in range(0, len(baseline), 32):
            n = min(32, len(baseline) - start)
            x.zero_(); b.zero_()
            x[:n].copy_(torch.from_numpy(np.array(observed[start:start+n], copy=True)))
            b[:n].copy_(torch.from_numpy(np.array(baseline[start:start+n], copy=True)))
            output.append(self(x, b)[:n].cpu().numpy())
        return np.concatenate(output)
