from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NLTLayerSpec:
    name: str
    weight_shape: tuple[int, ...]
    transform_shape: tuple[int, ...]
    groups: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NeuronLinearTransformConv2d(nn.Module):
    """Frozen Conv2d with a trainable neuron-pair affine weight transform.

    The effective kernel is ``source_weight * nlt_scale + nlt_shift``.  Scale
    and shift are shared over spatial kernel positions, matching the NLT
    definition while keeping every released-simulation source value frozen.
    """

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self.register_buffer("source_weight", source.weight.detach().clone())
        if source.bias is None:
            self.register_buffer("source_bias", None)
        else:
            self.register_buffer("source_bias", source.bias.detach().clone())
        shape = (source.out_channels, source.in_channels // source.groups, 1, 1)
        self.nlt_scale = nn.Parameter(
            torch.ones(shape, dtype=source.weight.dtype, device=source.weight.device)
        )
        self.nlt_shift = nn.Parameter(
            torch.zeros(shape, dtype=source.weight.dtype, device=source.weight.device)
        )

    def effective_weight(self) -> torch.Tensor:
        return self.source_weight * self.nlt_scale + self.nlt_shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            expanded = (padding[1], padding[1], padding[0], padding[0])
            inputs = F.pad(inputs, expanded, mode=self.padding_mode)
            padding = (0, 0)
        return F.conv2d(
            inputs,
            self.effective_weight(),
            self.source_bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


def apply_neuron_linear_transfer(module: nn.Module) -> list[NLTLayerSpec]:
    """Replace every descendant Conv2d in-place and return its frozen schema."""

    specs: list[NLTLayerSpec] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                transformed = NeuronLinearTransformConv2d(child)
                setattr(parent, name, transformed)
                specs.append(
                    NLTLayerSpec(
                        name=qualified,
                        weight_shape=tuple(child.weight.shape),
                        transform_shape=tuple(transformed.nlt_scale.shape),
                        groups=child.groups,
                    )
                )
            else:
                visit(child, qualified)

    visit(module, "")
    if not specs:
        raise ValueError("NLT conversion found no Conv2d layers")
    return specs


def named_nlt_parameters(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if name.endswith(".nlt_scale") or name.endswith(".nlt_shift"):
            yield name, parameter


def frozen_source_fingerprint(module: nn.Module) -> str:
    """Hash every non-NLT parameter and all frozen source buffers."""

    digest = hashlib.sha256()
    values: list[tuple[str, torch.Tensor]] = []
    values.extend(
        (name, value)
        for name, value in module.named_parameters()
        if not (name.endswith(".nlt_scale") or name.endswith(".nlt_shift"))
    )
    values.extend(
        (name, value)
        for name, value in module.named_buffers()
        if name.endswith(".source_weight") or name.endswith(".source_bias")
    )
    for name, value in sorted(values):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_complex():
            tensor = torch.view_as_real(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def nlt_diagnostics(module: nn.Module) -> dict[str, float | int]:
    scales: list[torch.Tensor] = []
    shifts: list[torch.Tensor] = []
    relative_updates: list[torch.Tensor] = []
    layers = 0
    for child in module.modules():
        if not isinstance(child, NeuronLinearTransformConv2d):
            continue
        layers += 1
        scales.append((child.nlt_scale.detach().float() - 1.0).reshape(-1))
        shifts.append(child.nlt_shift.detach().float().reshape(-1))
        source = child.source_weight.detach().float()
        delta = child.effective_weight().detach().float() - source
        relative_updates.append(
            torch.linalg.vector_norm(delta).reshape(1)
            / torch.linalg.vector_norm(source).clamp_min(1e-12).reshape(1)
        )
    scale = torch.cat(scales)
    shift = torch.cat(shifts)
    relative = torch.cat(relative_updates)
    return {
        "layers": layers,
        "parameters": int(scale.numel() + shift.numel()),
        "scale_delta_rms": float(scale.square().mean().sqrt()),
        "scale_delta_max_abs": float(scale.abs().max()),
        "shift_rms": float(shift.square().mean().sqrt()),
        "shift_max_abs": float(shift.abs().max()),
        "layer_relative_weight_update_mean": float(relative.mean()),
        "layer_relative_weight_update_max": float(relative.max()),
    }
