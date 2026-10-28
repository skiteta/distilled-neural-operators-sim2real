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
from factorized_model import FactorizedFlowOperator as HybridFlowOperator, ModelConfig

_ROOT = Path(__file__).resolve().parent
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERNAL_BATCH_SIZE = 3


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
from realpde.fused_half_ffn_cuda import FusedBiasFfn
_FUSED_BIAS_FFN = FusedBiasFfn(_DEVICE, _MODEL.blocks[0].ffn[0].in_channels, _MODEL.blocks[0].ffn[4].in_channels)
def _fixed_fused_bias_ffn(self, value):
    return _FUSED_BIAS_FFN(value, self[0].weight, self[0].bias, self[4].weight, self[4].bias)
for _block in _MODEL.blocks:
    _ffn = _block.ffn
    assert len(_ffn)==5 and not _ffn.training and isinstance(_ffn[1],torch.nn.GLU) and _ffn[1].dim==1 and isinstance(_ffn[2],torch.nn.GELU) and _ffn[2].approximate=="none" and isinstance(_ffn[3],torch.nn.Dropout)
    _ffn.forward = types.MethodType(_fixed_fused_bias_ffn, _ffn)


from realpde.fused_local_residual_cuda import FusedLocalResidual
from realpde.model import OperatorBlock
_FUSED_LOCAL_RESIDUAL = FusedLocalResidual(_DEVICE, bias_first=True, use_fma=True)
def _fixed_fused_local_forward(self, features):
    normalized = self.norm1(features)
    global_update = self.global_mix(normalized)
    channel_update = self.channel_mix(normalized)
    features = _FUSED_LOCAL_RESIDUAL(features, normalized, global_update, channel_update,
                                   self.local.weight, self.local.bias, self.residual_scale)
    return features + self.residual_scale * self.ffn(self.norm2(features))
_LOCAL_BLOCKS = [m for m in _MODEL.modules() if isinstance(m, OperatorBlock)]
assert len(_LOCAL_BLOCKS) == 12
for _block in _LOCAL_BLOCKS:
    assert _block.local.kernel_size == (5,5) and _block.local.padding == (2,2)
    assert _block.local.stride == (1,1) and _block.local.dilation == (1,1)
    assert _block.local.groups == _block.local.in_channels == _block.local.out_channels
    assert _block.global_mix is not None
    _block.forward = types.MethodType(_fixed_fused_local_forward, _block)

def _three_channels(value):
    return torch.cat((value, torch.zeros_like(value[..., :1])), dim=-1)


@torch.inference_mode()
def _tensor_forward(batch):
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
    return point, _three_channels(center - half_width), _three_channels(center + half_width), _three_channels(center)


class _GraphLane:
    def __init__(self, example):
        self.stream = torch.cuda.Stream(device=_DEVICE)
        self.stream.wait_stream(torch.cuda.current_stream(_DEVICE))
        with torch.cuda.stream(self.stream):
            self.input = example.clone()
            for _ in range(3):
                self.values = _tensor_forward(self.input)
        torch.cuda.current_stream(_DEVICE).wait_stream(self.stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.values = _tensor_forward(self.input)
        torch.cuda.current_stream(_DEVICE).wait_stream(self.stream)

    def cpu_values(self, batch, valid_count):
        self.stream.wait_stream(torch.cuda.current_stream(_DEVICE))
        with torch.cuda.stream(self.stream):
            self.input.copy_(batch)
            self.graph.replay()
            # Copies complete before reuse, so returned arrays never alias scratch.
            result = tuple(value[:valid_count].cpu() for value in self.values)
            self.input.zero_()
            for value in self.values:
                value.zero_()
        torch.cuda.current_stream(_DEVICE).wait_stream(self.stream)
        return result


# One permanent stream owns the graph. Lock protects scratch and lazy capture
# across host threads; no previous input or prediction participates in a replay.
import threading
_GRAPH_LOCK = threading.Lock()
_GRAPH_LANE = None


def _cpu_forward(batch, valid_count):
    global _GRAPH_LANE
    if _DEVICE.type != 'cuda':
        return tuple(value[:valid_count].cpu() for value in _tensor_forward(batch))
    with _GRAPH_LOCK:
        if _GRAPH_LANE is None:
            _GRAPH_LANE = _GraphLane(batch)
        return _GRAPH_LANE.cpu_values(batch, valid_count)



_TRANSFER_GROUP_SIZE = 48  # bounded GPU staging; a multiple of fixed network batch3

@torch.inference_mode()
def _cuda_predict(array, include_center):
    global _GRAPH_LANE
    keys = ["prediction", "lower", "upper"]
    if include_center:
        keys.append("adopted_center")
    pieces = {key: [] for key in keys}
    with _GRAPH_LOCK:
        for group_start in range(0, len(array), _TRANSFER_GROUP_SIZE):
            chunk = np.ascontiguousarray(array[group_start:group_start + _TRANSFER_GROUP_SIZE])
            valid = len(chunk)
            padded_count = ((valid + _INTERNAL_BATCH_SIZE - 1) // _INTERNAL_BATCH_SIZE) * _INTERNAL_BATCH_SIZE
            if padded_count != valid:
                padded = np.empty((padded_count, *chunk.shape[1:]), dtype=np.float32)
                padded[:valid] = chunk
                padded[valid:] = chunk[-1:]
                chunk = padded
            batch = torch.from_numpy(chunk).to(_DEVICE)
            if _GRAPH_LANE is None:
                _GRAPH_LANE = _GraphLane(batch[:_INTERNAL_BATCH_SIZE])
            lane = _GRAPH_LANE
            current = torch.cuda.current_stream(_DEVICE)
            lane.stream.wait_stream(current)
            with torch.cuda.stream(lane.stream):
                outputs = [torch.empty((valid, *value.shape[1:]), dtype=value.dtype, device=value.device) for value in lane.values[:len(keys)]]
                for start in range(0, padded_count, _INTERNAL_BATCH_SIZE):
                    count = min(_INTERNAL_BATCH_SIZE, valid - start)
                    lane.input.copy_(batch[start:start + _INTERNAL_BATCH_SIZE])
                    lane.graph.replay()
                    # All writes finish on this stream before scratch reuse.
                    for destination, value in zip(outputs, lane.values):
                        destination[start:start + count].copy_(value[:count])
                for key, value in zip(keys, outputs):
                    pieces[key].append(value.cpu())
                lane.input.zero_()
                for value in lane.values:
                    value.zero_()
            current.wait_stream(lane.stream)
    result = {key: torch.cat(values).numpy().astype(np.float32, copy=False) for key, values in pieces.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("model produced NaN or infinity")
    if not np.all(result["lower"] <= result["upper"]):
        raise RuntimeError("invalid prediction interval ordering")
    return result


@torch.inference_mode()
def _predict(input_array, *, include_center=False):
    array = np.asarray(input_array, dtype=np.float32)
    if array.ndim != 5 or array.shape[1:] != (20, 32, 64, 3):
        raise ValueError(f"expected [N,20,32,64,3], got {array.shape}")
    if len(array) == 0:
        raise ValueError("empty input batch")
    if _DEVICE.type == 'cuda':
        return _cuda_predict(array, include_center)
    predictions, lowers, uppers, centers = [], [], [], []
    for start in range(0, len(array), _INTERNAL_BATCH_SIZE):
        chunk = np.ascontiguousarray(array[start:start + _INTERNAL_BATCH_SIZE])
        valid_count = len(chunk)
        if valid_count < _INTERNAL_BATCH_SIZE:
            # Every network call has the same shape. Padding duplicates one
            # valid window; all computations are local to each window.
            padded = np.empty((_INTERNAL_BATCH_SIZE, *chunk.shape[1:]), dtype=np.float32)
            padded[:valid_count] = chunk
            padded[valid_count:] = chunk[-1:]
            chunk = padded
        batch = torch.from_numpy(chunk).to(_DEVICE)
        point, lower, upper, center = _cpu_forward(batch, valid_count)
        predictions.append(point)
        lowers.append(lower)
        uppers.append(upper)
        if include_center:
            centers.append(center)
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

# Asynchronous transfer changes only ownership/scheduling; all model values unchanged.
from async_transfer import install as _install_async_transfer
class _TransferNamespace:
    def __init__(self,namespace):object.__setattr__(self,'_namespace',namespace)
    def __getattr__(self,key):return self._namespace[key]
    def __setattr__(self,key,value):self._namespace[key]=value
_install_async_transfer(_TransferNamespace(globals()))
