"""Portable, unchanged C546 input-only spatial radius overlay.

No model loading, scoring, data access or process-wide numerical settings.
Training-only fit_ridge is exposed for the local runner; predict never calls it.
"""
from __future__ import annotations

import numpy as np
import torch

INPUT_ONLY_CHANNELS = tuple(list(range(9)) + list(range(16, 22)) + list(range(26, 66)) + list(range(106, 146)))
BANDS = ((0, 16), (16, 32), (32, 48), (48, 64))
FACTORS = np.asarray((.8, 1., 1.25), np.float32)
RIDGE_ALPHA = 10.


def input_features(diagnostics, scale):
    """95 CPU FP32 mean/std/absmean vectors plus two log scales; Torch std ddof1."""
    selected = diagnostics.detach().cpu()[:, INPUT_ONLY_CHANNELS].float()
    values = torch.cat((selected.mean((2, 3)), selected.std((2, 3)), selected.abs().mean((2, 3)),
                        scale.detach().cpu().float().reshape(len(scale), -1).log().clamp(-12., 12.)), dim=1)
    result = values.numpy()
    if result.shape != (len(scale), 287) or not np.isfinite(result).all():
        raise ValueError("nonfinite or malformed input-only features")
    return result


def fit_ridge(features, rewards):
    """C546 alpha10: FP64 training moments/solve, then frozen FP32 parameters."""
    x, y = np.asarray(features, np.float64), np.asarray(rewards, np.float64)
    if x.ndim != 2 or y.shape != (len(x), 8) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("invalid train-only ridge inputs")
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.
    normalized, y_mean = (x - mean) / std, y.mean(0)
    gram = normalized.T @ normalized
    gram.flat[::len(gram) + 1] += RIDGE_ALPHA
    weight = np.linalg.solve(gram, normalized.T @ (y - y_mean))
    result = {key: value.astype(np.float32) for key, value in
              (("mean", mean), ("std", std), ("y_mean", y_mean), ("weight", weight))}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite frozen ridge")
    return result


def predicted_gains(features, parameters):
    """Fixed FP32 B4 GEMM including an unpadded final batch, exactly as C546."""
    x = np.asarray(features, np.float32)
    result = np.empty((len(x), 8), np.float32)
    for start in range(0, len(x), 4):
        normalized = (x[start:start + 4] - parameters["mean"]) / parameters["std"]
        result[start:start + 4] = parameters["y_mean"] + normalized @ parameters["weight"]
    return result


def choose_actions(gains):
    """Per band, strict positive winner; exact tie or two nonpositives means 1."""
    values = np.asarray(gains, np.float32).reshape(-1, 4, 2)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite predicted reward")
    low, high = values[..., 0], values[..., 1]
    actions = np.ones(low.shape, np.int8)
    actions[(low > 0) & (low > high)] = 0
    actions[(high > 0) & (high > low)] = 2
    return actions


def routed_bounds(center, physical_radius, actions):
    """Apply factors to raw radius*scale; never reconstruct from rounded bounds."""
    actions = np.asarray(actions)
    if actions.shape != (len(center), 4) or not np.isin(actions, (0, 1, 2)).all():
        raise ValueError("invalid fixed band actions")
    factors = np.ones((len(center), 64), np.float32)
    for band, (left, right) in enumerate(BANDS):
        factors[:, left:right] = FACTORS[actions[:, band], None]
    factor = torch.from_numpy(factors).to(center.device).reshape(len(center), 1, 1, 64, 1)
    radius = physical_radius * factor
    return center - radius, center + radius
