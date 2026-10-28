"""Prediction intervals: four small heads that see only the current window.

    scale  = std of the observed u, v over the whole window (per window)
    shift  = center head + hidden head + observed-center head        (in units of `scale`)
    radius = radius head * exp(hidden head)                           (in units of `scale`)
    center = point + shift * scale,   lower/upper = center -/+ radius * scale

* center / radius heads read 186 parameter-free diagnostics of the observed window and the point forecast
  (`flow_diagnostics`); the radius head has two summed hidden streams (added capacity).
* the hidden head reads the point model's final hidden field.
* the observed-center head reads the hidden field plus the 40 time-centered observed u, v frames.
The heads were trained to maximize a smooth surrogate of the competition's Safe Prediction Score (see docs/pipeline.md).
"""

from __future__ import annotations

import torch
from torch import nn

from .model import coordinates


def _local_summaries(flow: torch.Tensor) -> torch.Tensor:
    temporal_mean = flow.mean(dim=1)
    fluctuation = flow - temporal_mean[:, None]
    variance = fluctuation.square().mean(dim=1)
    return torch.cat(
        (
            temporal_mean,
            variance.clamp_min(1e-8).sqrt(),
            0.5 * variance.sum(dim=-1, keepdim=True),
            (flow[:, -1] - flow[:, -5]) / 4.0,
            flow[:, -1] - temporal_mean,
        ),
        dim=-1,
    )


def _frequency_features(flow: torch.Tensor, bins: int) -> torch.Tensor:
    """Real and imaginary parts of the first `bins` nonzero temporal frequencies of the time-centered field."""
    spectrum = torch.fft.rfft((flow - flow.mean(dim=1, keepdim=True)).float(), dim=1, norm="ortho")
    used = min(bins, spectrum.shape[1] - 1)
    selected = spectrum[:, 1 : used + 1]
    if used < bins:
        selected = torch.cat((selected, selected.new_zeros(selected.shape[0], bins - used, *selected.shape[2:])), dim=1)
    return torch.view_as_real(selected).permute(0, 2, 3, 1, 4, 5).flatten(3)


def flow_diagnostics(observed: torch.Tensor, point: torch.Tensor, bins: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 186, H, W) diagnostics and the (N, 1, 1, 1, 1) window scale.

    Channels: local statistics of the observed window (11) and of the forecast (7), coordinates (2), window
    descriptors (6), all 40 normalized frames (80), and 10 temporal frequencies of both (80).
    """
    uv, point_uv = observed[..., :2].float(), point[..., :2].float()
    mean = uv.mean(dim=(1, 2, 3), keepdim=True)
    scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    x, p = (uv - mean) / scale, (point_uv - mean) / scale
    p_mean = p.mean(dim=1)
    p_variance = (p - p_mean[:, None]).square().mean(dim=1)
    point_summary = torch.cat(
        (p_mean, p_variance.clamp_min(1e-8).sqrt(), 0.5 * p_variance.sum(dim=-1, keepdim=True), p[:, -1] - x[:, -1]),
        dim=-1,
    )
    batch, _, height, width, _ = observed.shape
    grid = coordinates(batch, height, width, observed.device, observed.dtype).permute(0, 2, 3, 1)
    descriptors = torch.cat(
        (x[:, -1].mean(dim=(1, 2)), x[:, -1].std(dim=(1, 2)), p[:, -1].mean(dim=(1, 2)), p.std(dim=(1, 2, 3))), dim=-1
    )[:, None, None].expand(-1, height, width, -1)
    frames = torch.cat((x, p), dim=1).permute(0, 2, 3, 1, 4).flatten(3)
    frequency = torch.cat((_frequency_features(x, bins), _frequency_features(p, bins)), dim=-1)
    diagnostics = torch.cat((_local_summaries(x), point_summary, grid, descriptors, frames, frequency), dim=-1)
    return diagnostics.permute(0, 3, 1, 2).to(observed.dtype), scale


def _head_network(cin: int, hidden: int, out: int, bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, hidden, 1),
        nn.GELU(),
        nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
        nn.GELU(),
        nn.Conv2d(hidden, out, 1, bias=bias),
    )


class DiagnosticHead(nn.Module):
    """Per pixel and frame: shift = s_max * tanh(.), radius = exp(clamp(.)) for u and v, from the diagnostics."""

    def __init__(self, config: dict, two_streams: bool = False) -> None:
        super().__init__()
        self.config = config
        cin, hidden, out = config["diagnostic_channels"], config["hidden_channels"], config["out_steps"] * 4
        if two_streams:
            self.network = _TwoStreams(_head_network(cin, hidden, out), _head_network(cin, hidden, out, bias=False))
        else:
            self.network = _head_network(cin, hidden, out)

    def forward(self, diagnostics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(diagnostics)
        batch, _, height, width = raw.shape
        decisions = raw.reshape(batch, self.config["out_steps"], 2, 2, height, width)
        shift = torch.tanh(decisions[:, :, 0].permute(0, 1, 3, 4, 2)) * float(self.config["maximum_normalized_shift"])
        low = float(torch.log(torch.tensor(self.config["minimum_normalized_radius"])))
        high = float(torch.log(torch.tensor(self.config["maximum_normalized_radius"])))
        radius = decisions[:, :, 1].permute(0, 1, 3, 4, 2).clamp(min=low, max=high).exp()
        return shift, radius


class _TwoStreams(nn.Module):
    def __init__(self, base: nn.Module, extra: nn.Module) -> None:
        super().__init__()
        self.base, self.extra = base, extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.extra(x)


class HiddenResidualHead(nn.Module):
    """shift delta = 0.25 * tanh(.), log-radius delta = 0.5 * tanh(.), from a hidden field."""

    def __init__(self, channels: int, out_steps: int = 20) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.network = _head_network(channels, channels, out_steps * 4)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(hidden)
        batch, _, height, width = raw.shape
        value = raw.reshape(batch, self.out_steps, 2, 2, height, width)
        return 0.25 * torch.tanh(value[:, :, 0].permute(0, 1, 3, 4, 2)), 0.5 * torch.tanh(
            value[:, :, 1].permute(0, 1, 3, 4, 2)
        )


class IntervalHeads(nn.Module):
    def __init__(
        self,
        center: DiagnosticHead,
        radius: DiagnosticHead,
        hidden: HiddenResidualHead,
        observed_center: HiddenResidualHead,
    ) -> None:
        super().__init__()
        self.center, self.radius, self.hidden, self.observed_center = center, radius, hidden, observed_center

    def forward(
        self, observed: torch.Tensor, point: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """lower, upper for u and v: (N, 20, H, W, 2)."""
        diagnostics, scale = flow_diagnostics(observed, point)
        shift, _ = self.center(diagnostics.float())
        _, radius = self.radius(diagnostics.float())
        shift_delta, log_radius_delta = self.hidden(hidden)
        shift = shift + shift_delta
        radius = radius * log_radius_delta.exp()
        batch, steps, height, width, _ = observed.shape
        uv = observed[..., :2]
        frames = (
            ((uv - uv.mean(dim=1, keepdim=True)) / scale)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch, steps * 2, height, width)
        )
        observed_shift, _ = self.observed_center(torch.cat((hidden, frames), dim=1))
        shift = shift + observed_shift
        center = point[..., :2] + shift * scale
        half_width = radius * scale
        return center - half_width, center + half_width


def load_interval_heads(runtime_dir, device="cpu") -> IntervalHeads:
    def load(name):
        return torch.load(f"{runtime_dir}/{name}", map_location="cpu", weights_only=False)

    c, r, h, o = (load(n) for n in ("center_head.pt", "radius_head.pt", "hidden_head.pt", "observed_center_head.pt"))
    assert r.get("architecture") == "additive_capacity_2x"
    heads = IntervalHeads(
        DiagnosticHead(c["model_config"]),
        DiagnosticHead(r["model_config"], two_streams=True),
        HiddenResidualHead(**h["model_config"]),
        HiddenResidualHead(**o["model_config"]),
    )
    heads.center.load_state_dict(c["state_dict"], strict=True)
    heads.radius.load_state_dict(r["state_dict"], strict=True)
    heads.hidden.load_state_dict(h["state_dict"], strict=True)
    heads.observed_center.load_state_dict(o["state_dict"], strict=True)
    return heads.to(device).eval().requires_grad_(False)


__all__ = ["flow_diagnostics", "DiagnosticHead", "HiddenResidualHead", "IntervalHeads", "load_interval_heads"]
