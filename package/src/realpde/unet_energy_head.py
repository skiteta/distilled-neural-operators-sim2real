from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UNetEnergyHeadConfig:
    in_channels: int = 34
    width: int = 32
    dropout: float = 0.02
    maximum_log_amplitude: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "UNetEnergyHeadConfig":
        return cls(**values)


class Residual2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.block(values)


class UNetEnergyHead(nn.Module):
    def __init__(self, config: UNetEnergyHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetEnergyHeadConfig()
        w = self.config.width
        self.stem = nn.Conv2d(self.config.in_channels, w, 3, padding=1)
        self.enc0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.down0 = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.down1 = nn.Conv2d(2 * w, 4 * w, 4, stride=2, padding=1)
        self.middle = nn.Sequential(Residual2D(4 * w, self.config.dropout), Residual2D(4 * w, self.config.dropout))
        self.up1 = nn.Conv2d(6 * w, 2 * w, 3, padding=1)
        self.dec1 = nn.Sequential(Residual2D(2 * w, self.config.dropout), Residual2D(2 * w, self.config.dropout))
        self.up0 = nn.Conv2d(3 * w, w, 3, padding=1)
        self.dec0 = nn.Sequential(Residual2D(w, self.config.dropout), Residual2D(w, self.config.dropout))
        self.out = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> Tensor:
        x0 = self.enc0(self.stem(features))
        x1 = self.enc1(self.down0(x0))
        middle = self.middle(self.down1(x1))
        up1 = F.interpolate(middle, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(self.up1(torch.cat((up1, x1), dim=1)))
        up0 = F.interpolate(up1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        up0 = self.dec0(self.up0(torch.cat((up0, x0), dim=1)))
        return self.config.maximum_log_amplitude * torch.tanh(self.out(up0)[:, 0])


def local_energy(flow: Tensor) -> Tensor:
    uv = flow[..., :2].float()
    fluctuation = uv - uv.mean(dim=1, keepdim=True)
    return 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1)


def _band_fractions(flow: Tensor) -> list[Tensor]:
    uv = flow[..., :2].float()
    fluctuation = uv - uv.mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(fluctuation, dim=1, norm="ortho")
    fractions = []
    total = fluctuation.square().sum(dim=(1, 4)).clamp_min(1e-8)
    for start, stop in ((1, 3), (3, 6), (6, spectrum.shape[1])):
        retained = torch.zeros_like(spectrum)
        retained[:, start:stop] = spectrum[:, start:stop]
        band = torch.fft.irfft(retained, n=flow.shape[1], dim=1, norm="ortho")
        fractions.append(band.square().sum(dim=(1, 4)) / total)
    return fractions


def build_energy_features(observed: Tensor, anchor: Tensor, candidate: Tensor) -> Tensor:
    observed = observed[..., :2].float()
    anchor = anchor[..., :2].float()
    candidate = candidate[..., :2].float()
    observed_mean = observed.mean(dim=1)
    anchor_mean = anchor.mean(dim=1)
    candidate_mean = candidate.mean(dim=1)
    observed_energy = local_energy(observed)
    anchor_energy = local_energy(anchor)
    candidate_energy = local_energy(candidate)
    epsilon = 1e-8
    anchor_fluctuation = anchor - anchor_mean[:, None]
    candidate_fluctuation = candidate - candidate_mean[:, None]
    dot = (anchor_fluctuation * candidate_fluctuation).sum(dim=(1, 4))
    anchor_norm = anchor_fluctuation.square().sum(dim=(1, 4)).sqrt()
    candidate_norm = candidate_fluctuation.square().sum(dim=(1, 4)).sqrt()
    phase_cosine = dot / (anchor_norm * candidate_norm).clamp_min(epsilon)
    phase_disagreement = (anchor_fluctuation - candidate_fluctuation).square().sum(dim=(1, 4)).sqrt()
    phase_disagreement = phase_disagreement / (anchor_norm + candidate_norm).clamp_min(epsilon)
    dy = torch.zeros_like(observed_energy)
    dx = torch.zeros_like(observed_energy)
    dy[:, 1:-1] = 0.5 * (observed_mean[:, 2:, :, 0] - observed_mean[:, :-2, :, 0])
    dx[:, :, 1:-1] = 0.5 * (observed_mean[:, :, 2:, 1] - observed_mean[:, :, :-2, 1])
    vorticity = dx - dy
    speed = observed_mean.square().sum(dim=-1).sqrt()
    fluid = (observed.abs().sum(dim=(1, 4)) > 1e-7).float()
    height, width = observed.shape[2:4]
    y = torch.linspace(-1.0, 1.0, height, device=observed.device, dtype=observed.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=observed.device, dtype=observed.dtype)
    yy = y[None, :, None].expand(len(observed), height, width)
    xx = x[None, None, :].expand(len(observed), height, width)
    values = [
        observed_mean[..., 0], observed_mean[..., 1],
        observed[:, -1, ..., 0], observed[:, -1, ..., 1],
        observed[:, -1, ..., 0] - observed[:, -5, ..., 0],
        observed[:, -1, ..., 1] - observed[:, -5, ..., 1],
        anchor_mean[..., 0], anchor_mean[..., 1],
        candidate_mean[..., 0], candidate_mean[..., 1],
        candidate_mean[..., 0] - anchor_mean[..., 0],
        candidate_mean[..., 1] - anchor_mean[..., 1],
        torch.log(observed_energy.clamp_min(epsilon)),
        torch.log(anchor_energy.clamp_min(epsilon)),
        torch.log(candidate_energy.clamp_min(epsilon)),
        *_band_fractions(observed),
        *_band_fractions(anchor),
        *_band_fractions(candidate),
        torch.log((candidate_energy + epsilon) / (anchor_energy + epsilon)),
        torch.log((observed_energy + epsilon) / (anchor_energy + epsilon)),
        torch.log((candidate_energy + epsilon) / (observed_energy + epsilon)),
        phase_cosine,
        phase_disagreement,
        vorticity,
        speed,
        fluid,
        xx,
        yy,
    ]
    features = torch.stack(values, dim=1)
    if features.shape[1] != 34:
        raise RuntimeError(f"expected 34 energy features, got {features.shape[1]}")
    return features


def apply_energy_head(anchor: Tensor, phase: Tensor, log_amplitude: Tensor) -> Tensor:
    anchor = anchor[..., :2].float()
    phase = phase[..., :2].float()
    mean = anchor.mean(dim=1, keepdim=True)
    fluctuation = phase - phase.mean(dim=1, keepdim=True)
    phase_energy = local_energy(phase)
    target_energy = local_energy(anchor) * torch.exp(2.0 * log_amplitude)
    scale = (target_energy / phase_energy.clamp_min(1e-12)).sqrt()
    scaled = fluctuation * scale[:, None, :, :, None]
    scaled = torch.where(target_energy[:, None, :, :, None] > 1e-12, scaled, torch.zeros_like(scaled))
    return mean + scaled
