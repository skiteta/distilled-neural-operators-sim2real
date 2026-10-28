"""Window-local flow-shape head with differentiable mean/TKE preservation."""
import numpy as np
import torch
from torch import nn


def preserve_moments(candidate, baseline):
    """Project each cell's 20xUV trajectory, without detaching its gradient."""
    a, b = candidate.double(), baseline.double()
    mu = b.mean(1, keepdim=True)
    fa, fb = a - a.mean(1, keepdim=True), b - mu
    ea = fa.square().sum(-1, keepdim=True).mean(1, keepdim=True)
    eb = fb.square().sum(-1, keepdim=True).mean(1, keepdim=True)
    # Clamp both sides before sqrt to keep backward finite in stagnant cells.
    scale = (eb.clamp_min(1e-30) / ea.clamp_min(1e-30)).sqrt()
    corrected = torch.where((ea > 1e-30) & (eb > 1e-30), fa * scale, fb)
    return (mu + corrected).to(baseline.dtype)


class ShapeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 40, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observed, baseline, rolled=False):
        fo = observed - observed.mean(1, keepdim=True)
        fb = baseline - baseline.mean(1, keepdim=True)
        norm = (.5 * (fo.square().mean((1, 2, 3, 4), keepdim=True)
                      + fb.square().mean((1, 2, 3, 4), keepdim=True))).sqrt().clamp_min(1e-6)
        if rolled:
            fo = torch.roll(fo, (16, 32), (2, 3))
        pack = lambda x: x.permute(0, 1, 4, 2, 3).reshape(len(x), 40, x.shape[2], x.shape[3])
        z = self.net(torch.cat([pack(fo / norm), pack(fb / norm)], 1))
        z = z.reshape(len(z), 20, 2, z.shape[2], z.shape[3]).permute(0, 1, 3, 4, 2)
        local_rms = fb.square().sum(-1, keepdim=True).mean(1, keepdim=True).sqrt()
        candidate = baseline + .1 * torch.tanh(z) * local_rms
        return preserve_moments(candidate, baseline)

    @torch.inference_mode()
    def predict_numpy(self, observed, baseline, rolled=False, metadata=None):
        """Fixed one-window microbatch makes the public interface batch/order invariant."""
        if len(observed) != len(baseline):
            raise ValueError('observed and baseline lengths differ')
        if not len(baseline):
            return np.empty_like(baseline)
        device = next(self.parameters()).device
        out = []
        for x, b in zip(observed, baseline):
            xx = torch.from_numpy(np.array(x[None], copy=True)).to(device)
            bb = torch.from_numpy(np.array(b[None], copy=True)).to(device)
            out.append(self(xx, bb, rolled).cpu().numpy())
        return np.concatenate(out)
