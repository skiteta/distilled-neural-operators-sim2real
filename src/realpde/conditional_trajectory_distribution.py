"""C635 conditional whole-trajectory residual model and proper-score estimators.

No saved-run, target, metadata, or process-global random state is used at inference.
Scales are fitted by the caller from training cases only.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class ConditionalTrajectoryResidual(nn.Module):
    architecture = 'c635_raw20_baseline20_unet32_64_96_film32_temporal3'

    def __init__(self, input_mean=None, input_scale=None, residual_scale=None):
        super().__init__()
        for name, value, default in (
            ('input_mean', input_mean, [0., 0.]),
            ('input_scale', input_scale, [1., 1.]),
            ('residual_scale', residual_scale, [1., 1.]),
        ):
            self.register_buffer(name, torch.tensor(default if value is None else value, dtype=torch.float32))
        self.lift = nn.Conv2d(80, 32, 3, padding=1)
        self.enc0 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down2 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(96), ResidualBlock(96))
        self.film = nn.Linear(32, 192)
        self.up1 = nn.Conv2d(160, 64, 1)
        self.dec1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.up0 = nn.Conv2d(96, 32, 1)
        self.dec0 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.output = nn.Conv2d(32, 40, 1)
        self.temporal = nn.Conv3d(2, 2, (3, 1, 1), padding=(1, 0, 0), bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.temporal.weight)
        with torch.no_grad():
            self.temporal.weight[0, 0, 1, 0, 0] = 1
            self.temporal.weight[1, 1, 1, 0, 0] = 1

    def forward(self, observed, baseline, z):
        if observed.shape != baseline.shape or observed.shape[1:] != (20, 32, 64, 2):
            raise ValueError('Expected matching B,20,32,64,2 fields')
        if z.shape != (len(observed), 32):
            raise ValueError('Expected one 32-dimensional latent per whole trajectory')
        x = (observed - self.input_mean) / self.input_scale
        b = (baseline - self.input_mean) / self.input_scale
        packed = torch.cat([t.permute(0, 1, 4, 2, 3).reshape(-1, 40, 32, 64) for t in (x, b)], dim=1)
        e0 = self.enc0(self.lift(packed))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        gamma, beta = self.film(z).chunk(2, dim=1)
        e2 = e2 * (1 + .1 * gamma[:, :, None, None]) + .1 * beta[:, :, None, None]
        d1 = self.dec1(self.up1(torch.cat((F.interpolate(e2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1), 1)))
        d0 = self.dec0(self.up0(torch.cat((F.interpolate(d1, size=e0.shape[-2:], mode='bilinear', align_corners=False), e0), 1)))
        residual = self.output(d0).reshape(-1, 20, 2, 32, 64).permute(0, 2, 1, 3, 4)
        residual = self.temporal(residual).permute(0, 2, 3, 4, 1)
        # A fixed, observable mask only: never use future target geometry.
        mask = observed.abs().sum(dim=(1, 4)) > 0
        return baseline + residual * self.residual_scale * mask[:, None, :, :, None]


def content_latents(observed, count=16, seed=6341):
    """CPU latent bank independently keyed by each observed window's FP32 bytes."""
    values = observed.detach().cpu().contiguous().numpy().astype('<f4', copy=False)
    banks = []
    for x in values:
        digest = hashlib.sha256(str(seed).encode() + x.tobytes()).digest()
        rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63 - 1))
        banks.append(torch.randn(count, 32, generator=rng))
    return torch.stack(banks)


def sample_trajectories(model, observed, baseline, latents):
    batch, count, _ = latents.shape
    x = observed[:, None].expand(-1, count, -1, -1, -1, -1).reshape(batch * count, 20, 32, 64, 2)
    b = baseline[:, None].expand(-1, count, -1, -1, -1, -1).reshape_as(x)
    return model(x, b, latents.reshape(batch * count, 32)).reshape(batch, count, 20, 32, 64, 2)


def energy_score(samples, target):
    """Unbiased finite-ensemble ES, normalized by fixed sqrt(field dimension).

    K=1 is the point Euclidean score. torch.linalg norm defines a finite zero
    subgradient, including identical deterministic ensemble members.
    """
    flat = samples.flatten(2)
    truth = target.flatten(1)
    count = flat.shape[1]
    result = torch.linalg.vector_norm(flat - truth[:, None], dim=-1).mean(1)
    if count > 1:
        pair_sum = sum(torch.linalg.vector_norm(flat[:, k] - flat[:, j], dim=-1)
                       for k in range(count) for j in range(k))
        result = result - pair_sum / (count * (count - 1))
    return result / math.sqrt(flat.shape[-1])


def fixed_variogram_pairs(count=2048, seed=6341):
    """Balanced temporal lags1/5 and spatial nearest/stride4 pairs, same uv channel."""
    rng = np.random.default_rng(seed)
    result = []
    for n in range(count):
        kind = n % 4
        t, h, w, c = int(rng.integers(20)), int(rng.integers(32)), int(rng.integers(64)), int(rng.integers(2))
        tt, hh, ww = t, h, w
        if kind < 2:
            lag = (1, 5)[kind]
            t = int(rng.integers(20 - lag)); tt = t + lag
        else:
            lag = (1, 4)[kind - 2]
            if rng.integers(2):
                h = int(rng.integers(32 - lag)); hh = h + lag
            else:
                w = int(rng.integers(64 - lag)); ww = w + lag
        a = ((t * 32 + h) * 64 + w) * 2 + c
        b = ((tt * 32 + hh) * 64 + ww) * 2 + c
        result.append((a, b))
    return torch.tensor(result, dtype=torch.long)


def variogram_score(samples, target, pairs):
    """p=1 variogram score with an unbiased cross-member product estimator.

    Individual finite-sample estimates can be negative. No clipping is applied.
    """
    s, y = samples.flatten(2), target.flatten(1)
    a, b = pairs[:, 0], pairs[:, 1]
    error = (y[:, a] - y[:, b]).abs()[:, None] - (s[:, :, a] - s[:, :, b]).abs()
    count = s.shape[1]
    if count == 1:
        return error[:, 0].square().mean(1)
    return ((error.sum(1).square() - error.square().sum(1)) / (count * (count - 1))).mean(1)


def fair_crps(samples, target):
    count = samples.shape[1]
    first = (samples - target[:, None]).abs().flatten(2).mean(2).mean(1)
    if count == 1:
        return first
    pairs = sum((samples[:, k] - samples[:, j]).abs().flatten(1).mean(1)
                for k in range(count) for j in range(k))
    return first - pairs / (count * (count - 1))
