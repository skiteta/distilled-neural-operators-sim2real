"""Experimental identity bounds: original center head, independent radius head.

Packaged as submission.py with model.pt, center_head.pt and radius_head.pt.
No fitted calibration, router, cross-call state or process-wide setting changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from realpde.fast_interval import FastIntervalConfig, FastIntervalHead
from realpde.sps_radius_capacity import load_expanded_head
from realpde.hidden_sps_interval import load_hidden_residual, predict_with_hidden
from realpde.flow_diagnostics import flow_diagnostics
from realpde.model import HybridFlowOperator, ModelConfig

_ROOT = Path(__file__).resolve().parent
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERNAL_BATCH_SIZE = 32


def _load_flow_model(path):
    checkpoint = torch.load(path, map_location=_DEVICE, weights_only=False)
    model = HybridFlowOperator(ModelConfig.from_dict(checkpoint["model_config"])).to(_DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def _load_head(path):
    checkpoint = torch.load(path, map_location=_DEVICE, weights_only=False)
    model = FastIntervalHead(FastIntervalConfig.from_dict(checkpoint["model_config"])).to(_DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


_MODEL = _load_flow_model(_ROOT / "model.pt")
_CENTER_HEAD = _load_head(_ROOT / "center_head.pt")
_RADIUS_HEAD = load_expanded_head(_ROOT / "radius_head.pt", _DEVICE)
_HIDDEN_HEAD = load_hidden_residual(_ROOT / "hidden_head.pt", _DEVICE)
_OBSERVED_CENTER_HEAD = load_hidden_residual(_ROOT / "observed_center_head.pt", _DEVICE)



import types
from realpde.fused_bias_ffn_cuda import FusedBiasFfn
_FUSED_BIAS_FFN = FusedBiasFfn(_DEVICE, _MODEL.blocks[0].ffn[0].in_channels, _MODEL.blocks[0].ffn[4].in_channels)
def _fixed_fused_bias_ffn(self, value):
    return _FUSED_BIAS_FFN(value, self[0].weight, self[0].bias, self[4].weight, self[4].bias)
for _block in _MODEL.blocks:
    _ffn = _block.ffn
    assert len(_ffn)==5 and not _ffn.training and isinstance(_ffn[1],torch.nn.GLU) and _ffn[1].dim==1 and isinstance(_ffn[2],torch.nn.GELU) and _ffn[2].approximate=="none" and isinstance(_ffn[3],torch.nn.Dropout)
    _ffn.forward = types.MethodType(_fixed_fused_bias_ffn, _ffn)

def _three_channels(value):
    return torch.cat((value, torch.zeros_like(value[..., :1])), dim=-1)


@torch.inference_mode()
def _predict(input_array, *, include_center=False):
    array = np.asarray(input_array, dtype=np.float32)
    if array.ndim != 5 or array.shape[1:] != (20, 32, 64, 3):
        raise ValueError(f"expected [N,20,32,64,3], got {array.shape}")
    if len(array) == 0:
        raise ValueError("empty input batch")
    predictions, lowers, uppers, centers = [], [], [], []
    for start in range(0, len(array), _INTERNAL_BATCH_SIZE):
        batch = torch.from_numpy(np.ascontiguousarray(array[start:start + _INTERNAL_BATCH_SIZE])).to(_DEVICE)
        # Same full point call and batch size as protected C312.
        point, hidden = predict_with_hidden(_MODEL, batch)
        diagnostics, scale = flow_diagnostics(batch, point)
        shift, _ = _CENTER_HEAD(diagnostics.float())
        _, radius = _RADIUS_HEAD(diagnostics.float())
        shift_delta, log_radius_delta = _HIDDEN_HEAD(hidden)
        shift = shift + shift_delta
        radius = radius * log_radius_delta.exp()
        raw = ((batch[..., :2] - batch[..., :2].mean(dim=1, keepdim=True)) / scale).permute(0, 1, 4, 2, 3).reshape(-1, 40, 32, 64)
        observed_shift, _ = _OBSERVED_CENTER_HEAD(torch.cat((hidden, raw), dim=1))
        shift = shift + observed_shift
        # Exact identity arithmetic; neither checkpoint's optional settings enter.
        center = point[..., :2] + shift * scale
        half_width = radius * scale
        predictions.append(point.cpu())
        lowers.append(_three_channels(center - half_width).cpu())
        uppers.append(_three_channels(center + half_width).cpu())
        if include_center:
            centers.append(_three_channels(center).cpu())
    result = {"prediction": torch.cat(predictions).numpy().astype(np.float32, copy=False),
              "lower": torch.cat(lowers).numpy().astype(np.float32, copy=False),
              "upper": torch.cat(uppers).numpy().astype(np.float32, copy=False)}
    if include_center:
        result["adopted_center"] = torch.cat(centers).numpy().astype(np.float32, copy=False)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("model produced NaN or infinity")
    if not np.all(result["lower"] <= result["upper"]):
        raise RuntimeError("invalid prediction interval ordering")
    return result


def predict(input_array: np.ndarray, metadata=None):
    return _predict(input_array)


def predict_with_center(input_array: np.ndarray, metadata=None):
    """Validation-only extra output; normal predict keeps the official 3 keys."""
    return _predict(input_array, include_center=True)
