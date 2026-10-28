"""Bounded amplitude residual and train-only point non-regression penalties."""
import numpy as np
import torch
from realpde.baseline_matched_tke import LOG_CAP


def anchored_scales(reference, residual_scale):
    reference = reference.to(torch.float64)
    margin = (LOG_CAP - reference.log().abs()).clamp_min(0)
    direction = (residual_scale.to(torch.float64).log() / LOG_CAP).clamp(-1, 1)
    return reference * torch.exp(margin * direction)


def numpy_anchored_scales(reference, residual_scale):
    reference = np.asarray(reference, dtype=np.float64)
    margin = np.maximum(LOG_CAP - np.abs(np.log(reference)), 0)
    direction = np.clip(np.log(np.asarray(residual_scale, dtype=np.float64)) / LOG_CAP, -1, 1)
    return reference * np.exp(margin * direction)


def point_penalty(rel, tke, reference_rel, reference_tke):
    return ((rel-reference_rel)/reference_rel.clamp_min(1e-3)).relu() + ((tke-reference_tke)/reference_tke.clamp_min(1e-3)).relu()
