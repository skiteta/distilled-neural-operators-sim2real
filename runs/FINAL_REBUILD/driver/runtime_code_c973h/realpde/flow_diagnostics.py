from __future__ import annotations

import torch

from .model import HybridFlowOperator


def _local_summaries(flow: torch.Tensor) -> torch.Tensor:
    temporal_mean = flow.mean(dim=1)
    fluctuation = flow - temporal_mean[:, None]
    variance = fluctuation.square().mean(dim=1)
    temporal_std = variance.clamp_min(1e-8).sqrt()
    local_tke = 0.5 * variance.sum(dim=-1, keepdim=True)
    recent = (flow[:, -1] - flow[:, -5]) / 4.0
    final_fluctuation = flow[:, -1] - temporal_mean
    return torch.cat(
        (temporal_mean, temporal_std, local_tke, recent, final_fluctuation),
        dim=-1,
    )


def _frequency_features(flow: torch.Tensor, frequency_bins: int) -> torch.Tensor:
    centered = flow - flow.mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(centered.float(), dim=1, norm="ortho")
    bins = min(int(frequency_bins), spectrum.shape[1] - 1)
    selected = spectrum[:, 1 : bins + 1]
    if bins < frequency_bins:
        padding = selected.new_zeros(
            selected.shape[0],
            frequency_bins - bins,
            *selected.shape[2:],
        )
        selected = torch.cat((selected, padding), dim=1)
    packed = torch.view_as_real(selected)
    return packed.permute(0, 2, 3, 1, 4, 5).flatten(3)


def flow_diagnostics(
    observed: torch.Tensor,
    point: torch.Tensor,
    *,
    frequency_bins: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cheap 186-channel physical/frequency diagnostic tensor.

    The implementation is parameter-free and exactly matches the diagnostic
    construction used by ``SPSOrdinalUncertaintyOperator`` for 20 input/output
    steps and ten temporal-frequency bins.  Keeping it independent from the
    ordinal model avoids loading an unrelated point model at inference time.
    """

    if observed.ndim != 5 or point.ndim != 5:
        raise ValueError("observed and point must have shape (N,T,H,W,C)")
    if observed.shape[0] != point.shape[0] or observed.shape[2:4] != point.shape[2:4]:
        raise ValueError("observed and point batch/spatial shapes must match")
    uv = observed[..., :2].float()
    point_uv = point[..., :2].float()
    mean = uv.mean(dim=(1, 2, 3), keepdim=True)
    scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    normalized_input = (uv - mean) / scale
    normalized_point = (point_uv - mean) / scale
    input_summary = _local_summaries(normalized_input)
    point_mean = normalized_point.mean(dim=1)
    point_fluctuation = normalized_point - point_mean[:, None]
    point_variance = point_fluctuation.square().mean(dim=1)
    point_summary = torch.cat(
        (
            point_mean,
            point_variance.clamp_min(1e-8).sqrt(),
            0.5 * point_variance.sum(dim=-1, keepdim=True),
            normalized_point[:, -1] - normalized_input[:, -1],
        ),
        dim=-1,
    )
    batch, _, height, width, _ = observed.shape
    coordinates = HybridFlowOperator._coordinates(
        batch, height, width, observed.device, observed.dtype
    ).permute(0, 2, 3, 1)
    descriptors = torch.cat(
        (
            normalized_input[:, -1].mean(dim=(1, 2)),
            normalized_input[:, -1].std(dim=(1, 2)),
            normalized_point[:, -1].mean(dim=(1, 2)),
            normalized_point.std(dim=(1, 2, 3)),
        ),
        dim=-1,
    )
    descriptors = descriptors[:, None, None].expand(-1, height, width, -1)
    raw_phase = torch.cat((normalized_input, normalized_point), dim=1)
    raw_phase = raw_phase.permute(0, 2, 3, 1, 4).flatten(3)
    frequency = torch.cat(
        (
            _frequency_features(normalized_input, frequency_bins),
            _frequency_features(normalized_point, frequency_bins),
        ),
        dim=-1,
    )
    diagnostics = torch.cat(
        (
            input_summary,
            point_summary,
            coordinates,
            descriptors,
            raw_phase,
            frequency,
        ),
        dim=-1,
    )
    expected = 26 + 2 * (observed.shape[1] + point.shape[1]) + 8 * frequency_bins
    if diagnostics.shape[-1] != expected:
        raise RuntimeError(
            f"unexpected diagnostics: got {diagnostics.shape[-1]}, expected {expected}"
        )
    return diagnostics.permute(0, 3, 1, 2).to(observed.dtype), scale
