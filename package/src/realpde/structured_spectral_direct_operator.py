from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .structured_spectral_delay_operator import (
    StructuredSpectralDelayConfig,
    StructuredSpectralDelayOperator,
)


@dataclass(frozen=True)
class StructuredSpectralDirectConfig(StructuredSpectralDelayConfig):
    """Fixed C443 direct phase-preserving SSP synthesis configuration."""

    synthesis_width: int = 96
    correction_cap: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StructuredSpectralDirectConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredSpectralDirectOperator(StructuredSpectralDelayOperator):
    """Frozen C253 plus direct SSP-to-solenoidal-field residual synthesis."""

    def __init__(self, config: StructuredSpectralDirectConfig) -> None:
        super().__init__(config)
        self.operator_config = config
        del self.coefficient_head
        packed = 3 * config.latent_width * 2
        self.synthesis_in = nn.Conv2d(packed, config.synthesis_width, 1)
        self.synthesis_local = nn.Conv2d(
            config.synthesis_width,
            config.synthesis_width,
            3,
            padding=1,
            groups=config.synthesis_width,
            padding_mode="circular",
        )
        self.synthesis_out = nn.Conv2d(config.synthesis_width, 4, 1)
        nn.init.zeros_(self.synthesis_out.weight)
        nn.init.zeros_(self.synthesis_out.bias)

    def train(self, mode: bool = True) -> StructuredSpectralDirectOperator:
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _pack_direct_sources(
        propagated: torch.Tensor, point_spectrum: torch.Tensor
    ) -> torch.Tensor:
        sources = torch.cat(
            (propagated, point_spectrum, propagated - point_spectrum), dim=2
        )
        packed = torch.view_as_real(sources).permute(0, 1, 2, 5, 3, 4).contiguous()
        return packed.reshape(
            sources.shape[0] * sources.shape[1],
            sources.shape[2] * 2,
            sources.shape[-2],
            sources.shape[-1],
        )

    def _direct_synthesize(
        self,
        propagated: torch.Tensor,
        point_spectrum: torch.Tensor,
        scale: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = self._pack_direct_sources(propagated, point_spectrum)
        hidden = F.gelu(self.synthesis_in(packed))
        hidden = F.gelu(self.synthesis_local(hidden))
        raw = self.synthesis_out(hidden)
        batch, steps = propagated.shape[:2]
        modes_y = propagated.shape[-2]
        modes_x = propagated.shape[-1]
        raw = raw.reshape(batch, steps, 2, 2, modes_y, modes_x)
        raw = raw.permute(0, 1, 2, 4, 5, 3).contiguous()
        modes = torch.view_as_complex(raw.float())

        half = torch.zeros(
            batch,
            steps,
            2,
            height,
            width // 2 + 1,
            dtype=modes.dtype,
            device=modes.device,
        )
        my = self.operator_config.modes_y
        mx = self.operator_config.modes_x
        half[..., :my, :mx] = modes[..., :my, :mx]
        half[..., -my:, :mx] = modes[..., -my:, :mx]
        normalized = torch.fft.irfft2(
            half, s=(height, width), dim=(-2, -1), norm="ortho"
        ).permute(0, 1, 3, 4, 2)
        bounded = float(self.operator_config.correction_cap) * torch.tanh(normalized)
        correction = self._low_solenoidal(bounded * scale)
        return correction, raw

    @staticmethod
    def _relative_spectral_loss(
        correction: torch.Tensor, target_correction: torch.Tensor
    ) -> torch.Tensor:
        correction_hat = torch.fft.fft2(correction.float(), dim=(-3, -2), norm="ortho")
        target_hat = torch.fft.fft2(target_correction.float(), dim=(-3, -2), norm="ortho")
        numerator = (correction_hat - target_hat).abs().square().mean()
        denominator = target_hat.abs().square().mean().clamp_min(1e-8)
        return numerator / denominator

    def forward(
        self,
        flow: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        return_interval: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        if flow.ndim != 5 or tuple(flow.shape[1:]) != (20, 32, 64, 3):
            raise ValueError(f"expected [B,20,32,64,3], got {tuple(flow.shape)}")
        mean, scale = self._query_statistics(flow)
        observed_normalized = (flow[..., :2] - mean) / scale
        with torch.no_grad():
            base_prediction, base_half = self.base(flow, return_interval=True)
        point_normalized = (base_prediction[..., :2] - mean) / scale
        observed_spectrum = self._select_modes(
            self._encode_normalized(observed_normalized)
        )
        propagated, normality, orthogonality, closure = self._propagate(
            observed_spectrum
        )
        point_spectrum = self._select_modes(
            self._encode_normalized(point_normalized)
        )
        correction, synthesis_raw = self._direct_synthesize(
            propagated,
            point_spectrum,
            scale,
            height=flow.shape[2],
            width=flow.shape[3],
        )
        prediction_uv = base_prediction[..., :2] + correction
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )

        if return_details:
            details: dict[str, torch.Tensor] = {
                "prediction": prediction,
                "base_prediction": base_prediction,
                "base_half": base_half,
                "correction": correction,
                "propagated": propagated,
                "point_spectrum": point_spectrum,
                "normality": normality,
                "orthogonality": orthogonality,
                "closure": closure,
                "synthesis_raw": synthesis_raw,
            }
            if target is not None:
                target_normalized = (target[..., :2] - mean) / scale
                with torch.no_grad():
                    target_spectrum = self._select_modes(
                        self._encode_normalized(target_normalized)
                    )
                    target_correction = self._low_solenoidal(
                        target[..., :2] - base_prediction[..., :2]
                    )
                latent_loss = (propagated - target_spectrum).abs().square().mean()
                latent_loss = latent_loss / target_spectrum.abs().square().mean().clamp_min(1e-8)
                details["target_spectrum"] = target_spectrum
                details["target_correction"] = target_correction
                details["latent_loss"] = latent_loss
                details["residual_spectral_loss"] = self._relative_spectral_loss(
                    correction, target_correction
                )
            return details
        if return_interval:
            return prediction, base_half
        return prediction

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)

    @torch.inference_mode()
    def diagnostics(self, flow: torch.Tensor) -> dict[str, Any]:
        details = self.forward(flow, return_details=True)
        assert isinstance(details, dict)
        values = (
            details["prediction"], details["correction"], details["propagated"],
            details["closure"], details["synthesis_raw"],
        )
        return {
            "correction_rms": float(details["correction"].square().mean().sqrt()),
            "correction_abs_max": float(details["correction"].abs().max()),
            "synthesis_rms": float(details["synthesis_raw"].square().mean().sqrt()),
            "closure_rms": float(details["closure"].abs().square().mean().sqrt()),
            "normality": float(details["normality"]),
            "orthogonality": float(details["orthogonality"]),
            "all_finite": all(bool(torch.isfinite(value).all()) for value in values),
        }
