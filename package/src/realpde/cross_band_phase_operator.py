from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import HybridFlowOperator, LowRankSpectralMix2d, ModelConfig


@dataclass(frozen=True)
class CrossBandPhaseOperatorConfig:
    """Frozen C429 jointly learned cross-band residual operator."""

    base: dict[str, Any]
    branch_width: int = 64
    spectral_rank: int = 12
    modes_y: int = 12
    modes_x: int = 20
    residual_cap: float = 0.20
    phase_destroyed: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CrossBandPhaseOperatorConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossBandPhaseOperator(nn.Module):
    """Complete Hybrid future plus one horizon-shared phase-covariant operator."""

    def __init__(self, config: CrossBandPhaseOperatorConfig) -> None:
        super().__init__()
        self.operator_config = config
        base_config = ModelConfig.from_dict(config.base)
        self.config = base_config
        self.base = HybridFlowOperator(base_config)
        branch_width = int(config.branch_width)
        groups = 8 if branch_width % 8 == 0 else 1
        self.branch_lift = nn.Conv2d(32, branch_width, 1)
        self.branch_norm1 = nn.GroupNorm(groups, branch_width)
        self.branch_local = nn.Conv2d(
            branch_width,
            branch_width,
            3,
            padding=1,
            groups=branch_width,
            padding_mode="circular",
        )
        self.branch_channel = nn.Conv2d(branch_width, branch_width, 1)
        self.branch_spectral = LowRankSpectralMix2d(
            branch_width,
            int(config.spectral_rank),
            int(config.modes_y),
            int(config.modes_x),
        )
        self.branch_norm2 = nn.GroupNorm(groups, branch_width)
        self.branch_ffn = nn.Sequential(
            nn.Conv2d(branch_width, branch_width * 4, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Conv2d(branch_width * 2, branch_width, 1),
        )
        self.branch_scale = nn.Parameter(
            torch.full((1, branch_width, 1, 1), 1e-2)
        )
        self.branch_head = nn.Sequential(
            nn.Conv2d(
                branch_width,
                branch_width,
                3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv2d(branch_width, 2, 1),
        )
        nn.init.zeros_(self.branch_head[-1].weight)
        nn.init.zeros_(self.branch_head[-1].bias)

        directions = torch.tensor(
            [(1.0, float(s2), float(s3), float(s4)) for s2, s3, s4 in product((-1, 1), repeat=3)],
            dtype=torch.float32,
        )
        self.register_buffer("directions", directions, persistent=True)

    def _shared(
        self, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self.base._features(flow)
        hidden = self.base.lift(features)
        for index, block in enumerate(self.base.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.base.blocks) // 2):
                hidden = self.base.multiscale(hidden)
        hidden = self.base.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.base.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        normalized_last = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        normalized_recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (
            4.0 * uv_std
        )
        trend = normalized_recent * self.base.trend_scale[None, :, None, None, :]
        common = (normalized_last + trend + correction) * uv_std + uv_mean
        return hidden, common, uv_mean, uv_std

    @staticmethod
    def _masks(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        iy = torch.arange(height, device=device)
        ix = torch.arange(width, device=device)
        sy = torch.minimum(iy, height - iy).float()
        sx = torch.minimum(ix, width - ix).float()
        q = torch.sqrt(sy[:, None].square() + sx[None, :].square())
        low = (q > 0) & (q <= 4)
        meso = (q > 4) & (q <= 12)
        return low, meso

    @staticmethod
    def _filter(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(field.float(), dim=(-3, -2))
        filtered = spectrum * mask[None, None, :, :, None]
        return torch.fft.ifft2(filtered, dim=(-3, -2)).real.to(field.dtype)

    @staticmethod
    def query_shifts(flow: torch.Tensor) -> torch.Tensor:
        """Nonzero, translation-invariant per-query shifts for the paired control."""
        height, width = flow.shape[-3:-1]
        u_moment = flow[..., 0].abs().double().mean(dim=(1, 2, 3))
        v_moment = flow[..., 1].abs().double().mean(dim=(1, 2, 3))
        dy = torch.remainder(torch.floor(u_moment * 1_000_003).long(), height - 1) + 1
        dx = torch.remainder(torch.floor(v_moment * 1_000_033).long(), width - 1) + 1
        return torch.stack((dy, dx), dim=-1)

    @staticmethod
    def _roll_per_sample(field: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        batch, horizons, height, width, channels = field.shape
        y_index = torch.remainder(
            torch.arange(height, device=field.device)[None, :] - shifts[:, :1], height
        )
        rolled = torch.gather(
            field,
            2,
            y_index[:, None, :, None, None].expand(
                batch, horizons, height, width, channels
            ),
        )
        x_index = torch.remainder(
            torch.arange(width, device=field.device)[None, :] - shifts[:, 1:], width
        )
        return torch.gather(
            rolled,
            3,
            x_index[:, None, None, :, None].expand(
                batch, horizons, height, width, channels
            ),
        )

    def branch_features(
        self, flow: torch.Tensor, common: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizons, height, width, _ = common.shape
        low_mask, meso_mask = self._masks(height, width, common.device)
        point_bands = (
            self._filter(common, low_mask),
            self._filter(common, meso_mask),
        )
        observed = flow[:, -1:, ..., :2]
        observed_bands = [
            self._filter(observed, low_mask),
            self._filter(observed, meso_mask),
        ]
        shifts = self.query_shifts(flow)
        if self.operator_config.phase_destroyed:
            observed_bands = [
                self._roll_per_sample(value, shifts) for value in observed_bands
            ]
        projections: list[torch.Tensor] = []
        directions = self.directions.to(dtype=common.dtype)
        for point_index, observed_index in ((0, 0), (0, 1), (1, 0), (1, 1)):
            state = torch.cat(
                (
                    point_bands[point_index],
                    observed_bands[observed_index].expand(-1, horizons, -1, -1, -1),
                ),
                dim=-1,
            )
            projection = torch.einsum("bthwc,dc->btdhw", state, directions)
            rms = projection.float().square().mean(dim=(-2, -1), keepdim=True).sqrt()
            projections.append(projection / (rms.to(projection.dtype) + 1e-6))
        return torch.cat(projections, dim=2), shifts

    def _branch_raw(self, features: torch.Tensor) -> torch.Tensor:
        batch, horizons, channels, height, width = features.shape
        latent = self.branch_lift(features.reshape(batch * horizons, channels, height, width))
        normalized = self.branch_norm1(latent)
        update = (
            self.branch_local(normalized)
            + self.branch_channel(normalized)
            + self.branch_spectral(normalized)
        )
        latent = latent + self.branch_scale * update
        latent = latent + self.branch_scale * self.branch_ffn(self.branch_norm2(latent))
        return self.branch_head(latent).reshape(batch, horizons, 2, height, width).permute(
            0, 1, 3, 4, 2
        )

    def forward(
        self,
        flow: torch.Tensor,
        *,
        return_interval: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        hidden, common, _uv_mean, uv_std = self._shared(flow)
        features, shifts = self.branch_features(flow, common)
        raw_residual = self._branch_raw(features)
        residual = (
            float(self.operator_config.residual_cap)
            * torch.tanh(raw_residual)
            * uv_std
        )
        prediction_uv = common + residual
        pressure = torch.zeros_like(prediction_uv[..., :1])
        prediction = torch.cat((prediction_uv, pressure), dim=-1)

        batch, _, height, width = hidden.shape
        raw_width = self.base.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.base.interval_temperature.clamp_min(1e-3)
        )
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        if return_details:
            return {
                "prediction": prediction,
                "half_width": half,
                "common": torch.cat((common, torch.zeros_like(common[..., :1])), dim=-1),
                "features": features,
                "raw_residual": raw_residual,
                "residual": residual,
                "shifts": shifts,
            }
        if return_interval:
            return prediction, half
        return prediction

    @torch.inference_mode()
    def diagnostics(self, flow: torch.Tensor) -> dict[str, Any]:
        details = self(flow, return_details=True)
        features = details["features"].float()
        residual = details["residual"].float()
        shifts = details["shifts"].long()
        return {
            "feature_shape": list(features.shape),
            "feature_rms": float(features.square().mean().sqrt()),
            "feature_std": float(features.std()),
            "feature_finite": bool(torch.isfinite(features).all()),
            "residual_rms": float(residual.square().mean().sqrt()),
            "residual_max": float(residual.abs().max()),
            "shifts": shifts.cpu().tolist(),
            "all_shifts_nonzero": bool((shifts > 0).all()),
            "unique_shifts": len({tuple(value) for value in shifts.cpu().tolist()}),
        }

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self(input_array)
