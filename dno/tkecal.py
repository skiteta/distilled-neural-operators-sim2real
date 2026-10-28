"""TKECAL: turbulent-kinetic-energy calibration of the forecast, from the same window only.

The point forecast regresses toward the conditional mean, so its fluctuations around the temporal mean are often too
weak (or too strong) compared with the turbulence actually present. TKECAL rescales them per pixel toward the
fluctuation energy of the *observed* frames:

    p' = m + r * (p - m),   m = temporal mean of the 20 forecast frames
    r  = clip((G * k_obs / max(G * k_pred, 1e-12)) ** (gamma / 2), 0.8, 1.25),   gamma = 0.1
    k  = 0.5 * sum over u, v of the temporal (population) variance of the 20 observed / forecast frames
    G  = separable Gaussian blur, sigma 4 px, radius 16, edge-replicate padding

Pressure and the prediction intervals are unchanged (the intervals are built from the uncalibrated forecast).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

GAMMA = 0.10
SIGMA = 4.0
RADIUS = int(4.0 * SIGMA + 0.5)  # 16, as scipy.ndimage.gaussian_filter with truncate=4
RATIO_RANGE = (0.8, 1.25)


def gaussian_kernel(device=None) -> torch.Tensor:
    w = torch.exp(-0.5 * (torch.arange(-RADIUS, RADIUS + 1, dtype=torch.float64) / SIGMA) ** 2)
    return (w / w.sum()).to(device=device, dtype=torch.float32)


def blur(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of (N, H, W) with edge-replicate padding."""
    g = field[:, None]
    g = F.conv2d(F.pad(g, (RADIUS, RADIUS, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    g = F.conv2d(F.pad(g, (0, 0, RADIUS, RADIUS), mode="replicate"), kernel.view(1, 1, -1, 1))
    return g[:, 0]


def tke_calibrate(observed: torch.Tensor, point: torch.Tensor, kernel: torch.Tensor | None = None) -> torch.Tensor:
    kernel = gaussian_kernel(point.device) if kernel is None else kernel
    k_obs = 0.5 * observed[..., :2].var(dim=1, unbiased=False).sum(-1)
    uv = point[..., :2]
    mean = uv.mean(dim=1, keepdim=True)
    deviation = uv - mean
    k_pred = 0.5 * deviation.square().mean(dim=1).sum(-1)
    ratio = (blur(k_obs, kernel) / blur(k_pred, kernel).clamp_min(1e-12)).pow(GAMMA / 2).clamp(*RATIO_RANGE)
    return torch.cat((mean + ratio[:, None, :, :, None] * deviation, point[..., 2:]), dim=-1)
