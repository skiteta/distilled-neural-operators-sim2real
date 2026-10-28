from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import HybridFlowOperator, ModelConfig


@dataclass(frozen=True)
class StructuredSpectralDelayConfig:
    """Frozen C442 propagation-state and C431 point-delay synthesis config."""

    base: dict[str, Any]
    in_steps: int = 20
    out_steps: int = 20
    analysis_width: int = 32
    latent_width: int = 12
    modes_y: int = 12
    modes_x: int = 20
    closure_width: int = 96
    propagation_substeps: int = 2
    propagation_step: float = 0.5
    frequency_gate_scale: float = 0.25
    coefficient_cap: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StructuredSpectralDelayConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredSpectralDelayOperator(nn.Module):
    """Frozen C253 plus an SSP-conditioned C431 point-delay correction."""

    band_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    delay_indices = (19, 18, 15, 10, 0)
    atom_count = 88

    def __init__(self, config: StructuredSpectralDelayConfig) -> None:
        super().__init__()
        self.operator_config = config
        if config.in_steps != 20 or config.out_steps != 20:
            raise ValueError("C442 is fixed to the official 20-to-20 contract")
        if config.latent_width != 12 or config.modes_y != 12 or config.modes_x != 20:
            raise ValueError("C442 latent width and retained modes are frozen")
        self.config = ModelConfig.from_dict(config.base)
        self.base = HybridFlowOperator(self.config)
        self.base.requires_grad_(False)

        self.frame_lift = nn.Conv2d(4, config.analysis_width, 1)
        self.frame_project = nn.Conv2d(config.analysis_width, config.latent_width, 1)
        temporal = torch.eye(config.in_steps).repeat(config.latent_width, 1, 1)
        temporal = temporal + 0.01 * torch.randn_like(temporal)
        self.temporal_backbone = nn.Parameter(temporal)
        self.frequency_gate = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, config.latent_width),
        )

        packed = config.in_steps * config.latent_width * 2
        self.closure_in = nn.Conv2d(packed, config.closure_width, 1)
        self.closure_local = nn.Conv2d(
            config.closure_width,
            config.closure_width,
            3,
            padding=1,
            groups=config.closure_width,
            padding_mode="circular",
        )
        self.closure_out = nn.Conv2d(config.closure_width, packed, 1)
        nn.init.zeros_(self.closure_out.weight)
        nn.init.zeros_(self.closure_out.bias)

        summary_width = 16 * config.latent_width
        self.coefficient_head = nn.Sequential(
            nn.Linear(summary_width, 192),
            nn.GELU(),
            nn.Linear(192, self.atom_count),
        )
        nn.init.zeros_(self.coefficient_head[-1].weight)
        nn.init.zeros_(self.coefficient_head[-1].bias)

        frequency_features = self._frequency_features(config.modes_y, config.modes_x)
        self.register_buffer("frequency_features", frequency_features, persistent=True)

    def train(self, mode: bool = True) -> StructuredSpectralDelayOperator:
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _coordinates(
        batch: int,
        steps: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid[None, None].expand(batch, steps, -1, -1, -1)

    @staticmethod
    def _frequency_features(modes_y: int, modes_x: int) -> torch.Tensor:
        ky = torch.cat((torch.arange(modes_y), torch.arange(-modes_y, 0))).float()
        kx = torch.arange(modes_x).float()
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        norm = torch.sqrt(grid_x.square() + grid_y.square())
        theta = torch.atan2(grid_y, grid_x.clamp_min(1e-12))
        scale = float(max(modes_y, modes_x - 1, 1))
        return torch.stack(
            (grid_x / scale, grid_y / scale, norm / scale, torch.cos(theta), torch.sin(theta)),
            dim=-1,
        )

    @staticmethod
    def _query_statistics(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = flow[..., :2]
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return mean, std

    def _encode_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width, channels = normalized.shape
        if channels != 2:
            raise ValueError("C442 analysis expects normalized u/v")
        coordinates = self._coordinates(
            batch, steps, height, width, normalized.device, normalized.dtype
        )
        frames = torch.cat((normalized, coordinates), dim=-1)
        frames = frames.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, 4, height, width
        )
        hidden = F.gelu(self.frame_lift(frames))
        latent = self.frame_project(hidden)
        return latent.reshape(
            batch, steps, self.operator_config.latent_width, height, width
        )

    def _select_modes(self, latent: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(latent.float(), dim=(-2, -1), norm="ortho")
        my = self.operator_config.modes_y
        mx = self.operator_config.modes_x
        return torch.cat(
            (spectrum[..., :my, :mx], spectrum[..., -my:, :mx]), dim=-2
        )

    @staticmethod
    def _pack_complex(value: torch.Tensor) -> torch.Tensor:
        packed = torch.view_as_real(value).permute(0, 1, 2, 5, 3, 4).contiguous()
        return packed.reshape(value.shape[0], -1, value.shape[-2], value.shape[-1])

    def _unpack_complex(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        packed = value.reshape(
            batch,
            self.operator_config.in_steps,
            self.operator_config.latent_width,
            2,
            height,
            width,
        ).permute(0, 1, 2, 4, 5, 3).contiguous()
        return torch.view_as_complex(packed.float())

    def _closure(self, spectrum: torch.Tensor) -> torch.Tensor:
        packed = self._pack_complex(spectrum)
        hidden = F.gelu(self.closure_in(packed))
        hidden = F.gelu(self.closure_local(hidden))
        return self._unpack_complex(self.closure_out(hidden))

    def _linear_backbone(self, spectrum: torch.Tensor) -> torch.Tensor:
        advanced = torch.einsum(
            "btcyx,cot->bocyx",
            spectrum,
            self.temporal_backbone.to(spectrum.dtype),
        )
        gate = self.frequency_gate(self.frequency_features).permute(2, 0, 1)
        multiplier = 1.0 + float(self.operator_config.frequency_gate_scale) * torch.tanh(gate)
        return advanced * multiplier[None, None]

    def _propagate(
        self, spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = spectrum
        final_linear_delta = torch.zeros_like(state)
        final_closure = torch.zeros_like(state)
        for _ in range(int(self.operator_config.propagation_substeps)):
            linear = self._linear_backbone(state)
            final_linear_delta = linear - state
            final_closure = self._closure(state)
            state = state + float(self.operator_config.propagation_step) * (
                final_linear_delta + final_closure
            )

        matrix = self.temporal_backbone
        commutator = matrix @ matrix.transpose(-1, -2) - matrix.transpose(-1, -2) @ matrix
        normality = commutator.square().mean()
        linear_real = torch.view_as_real(final_linear_delta)
        closure_real = torch.view_as_real(final_closure)
        dot = (linear_real * closure_real).sum()
        linear_norm = (linear_real.square().sum() + 1e-8).sqrt()
        closure_norm = (closure_real.square().sum() + 1e-8).sqrt()
        orthogonality = (dot / (linear_norm * closure_norm)).square()
        return state, normality, orthogonality, final_closure

    @staticmethod
    def _spectral_summary(value: torch.Tensor) -> torch.Tensor:
        real = value.real
        imaginary = value.imag
        amplitude = value.abs()
        dims = (-2, -1)
        return torch.cat(
            (
                real.mean(dim=dims),
                real.std(dim=dims),
                imaginary.mean(dim=dims),
                imaginary.std(dim=dims),
                amplitude.mean(dim=dims),
                amplitude.std(dim=dims),
                amplitude.square().mean(dim=dims).sqrt(),
                amplitude.amax(dim=dims),
            ),
            dim=-1,
        )

    def _coefficients(
        self, propagated: torch.Tensor, point_spectrum: torch.Tensor
    ) -> torch.Tensor:
        summaries = torch.cat(
            (self._spectral_summary(propagated), self._spectral_summary(point_spectrum)),
            dim=-1,
        )
        blocks = summaries.reshape(summaries.shape[0], 4, 5, -1).mean(dim=2)
        raw = self.coefficient_head(blocks)
        return float(self.operator_config.coefficient_cap) * torch.tanh(raw)

    @staticmethod
    def _band_masks(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        iy = torch.arange(height, device=device)
        ix = torch.arange(width, device=device)
        sy = torch.minimum(iy, height - iy).float()
        sx = torch.minimum(ix, width - ix).float()
        q = torch.sqrt(sy[:, None].square() + sx[None, :].square())
        return (q > 0) & (q <= 4), (q > 4) & (q <= 12)

    @staticmethod
    def _split_bands(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = value.shape[-3:-1]
        low, meso = StructuredSpectralDelayOperator._band_masks(height, width, value.device)
        spectrum = torch.fft.fft2(value.float(), dim=(-3, -2))
        low_value = torch.fft.ifft2(
            spectrum * low[None, None, :, :, None], dim=(-3, -2)
        ).real
        meso_value = torch.fft.ifft2(
            spectrum * meso[None, None, :, :, None], dim=(-3, -2)
        ).real
        return low_value, meso_value

    @staticmethod
    def _split_bands_and_derivatives(
        value: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
        """Reuse one FFT for the C431 bands and their two periodic derivatives."""
        height, width = value.shape[-3:-1]
        low, meso = StructuredSpectralDelayOperator._band_masks(
            height, width, value.device
        )
        spectrum = torch.fft.fft2(value.float(), dim=(-3, -2))
        fy = 2.0 * torch.pi * torch.fft.fftfreq(
            height, device=value.device, dtype=value.dtype
        )
        fx = 2.0 * torch.pi * torch.fft.fftfreq(
            width, device=value.device, dtype=value.dtype
        )
        multipliers = (
            (1j * fy)[None, None, :, None, None],
            (1j * fx)[None, None, None, :, None],
        )
        bands: list[torch.Tensor] = []
        derivatives: dict[tuple[int, int], torch.Tensor] = {}
        for band_index, mask in enumerate((low, meso)):
            selected = spectrum * mask[None, None, :, :, None]
            bands.append(torch.fft.ifft2(selected, dim=(-3, -2)).real)
            for axis, multiplier in enumerate(multipliers):
                derivatives[(band_index, axis)] = torch.fft.ifft2(
                    selected * multiplier, dim=(-3, -2)
                ).real
        return (bands[0], bands[1]), derivatives

    @staticmethod
    def _derivative(value: torch.Tensor, axis: int) -> torch.Tensor:
        height, width = value.shape[-3:-1]
        spectrum = torch.fft.fft2(value.float(), dim=(-3, -2))
        if axis == 0:
            frequency = 2.0 * torch.pi * torch.fft.fftfreq(
                height, device=value.device, dtype=value.dtype
            )
            multiplier = (1j * frequency)[None, None, :, None, None]
        elif axis == 1:
            frequency = 2.0 * torch.pi * torch.fft.fftfreq(
                width, device=value.device, dtype=value.dtype
            )
            multiplier = (1j * frequency)[None, None, None, :, None]
        else:
            raise ValueError(axis)
        return torch.fft.ifft2(spectrum * multiplier, dim=(-3, -2)).real

    @staticmethod
    def _low_solenoidal(value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-3:-1]
        u_hat = torch.fft.fft2(value[..., 0], dim=(-2, -1))
        v_hat = torch.fft.fft2(value[..., 1], dim=(-2, -1))
        ky = torch.fft.fftfreq(height, dtype=value.dtype, device=value.device)
        kx = torch.fft.fftfreq(width, dtype=value.dtype, device=value.device)
        ky = ky.clone()
        kx = kx.clone()
        ky[height // 2] = 0
        kx[width // 2] = 0
        derivative_ky, derivative_kx = torch.meshgrid(ky, kx, indexing="ij")
        k2 = derivative_kx.square() + derivative_ky.square()
        safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
        dot = derivative_kx * u_hat + derivative_ky * v_hat
        sol_u = u_hat - derivative_kx * dot / safe
        sol_v = v_hat - derivative_ky * dot / safe
        iy = torch.arange(height, device=value.device)
        ix = torch.arange(width, device=value.device)
        sy = torch.minimum(iy, height - iy).float()
        sx = torch.minimum(ix, width - ix).float()
        mask = torch.sqrt(sy[:, None].square() + sx[None, :].square()) <= 12
        sol_u = sol_u * mask
        sol_v = sol_v * mask
        return torch.stack(
            (
                torch.fft.ifft2(sol_u, dim=(-2, -1)).real,
                torch.fft.ifft2(sol_v, dim=(-2, -1)).real,
            ),
            dim=-1,
        )

    @classmethod
    def atom_order(cls) -> tuple[str, ...]:
        names: list[str] = []
        for first, second in cls.band_pairs:
            for axis in (0, 1):
                names.append(f"point_self:p{first}->p{second}:axis{axis}")
        delay_names = ("O19", "D1", "D4", "D9", "D19")
        for state_name in delay_names:
            for point_band, state_band in cls.band_pairs:
                for order in ("point_state", "state_point"):
                    for axis in (0, 1):
                        names.append(
                            f"point_delay:{state_name}:p{point_band}:s{state_band}:"
                            f"{order}:axis{axis}"
                        )
        if len(names) != cls.atom_count:
            raise RuntimeError("C442 atom-order length mismatch")
        return tuple(names)

    def _point_delay_basis(
        self,
        observed_normalized: torch.Tensor,
        point_normalized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            observed = observed_normalized - observed_normalized.mean(
                dim=(2, 3), keepdim=True
            )
            point = point_normalized - point_normalized.mean(dim=(2, 3), keepdim=True)
            current = observed[:, 19]
            states = torch.stack(
                (
                    current,
                    current - observed[:, 18],
                    (current - observed[:, 15]) / 4.0,
                    (current - observed[:, 10]) / 9.0,
                    (current - observed[:, 0]) / 19.0,
                ),
                dim=1,
            )
            point_bands, point_derivatives = self._split_bands_and_derivatives(point)
            state_stacks, stacked_state_derivatives = self._split_bands_and_derivatives(
                states
            )
            state_bands = tuple(
                tuple(band[:, index : index + 1] for band in state_stacks)
                for index in range(5)
            )
            state_derivatives = {
                (state, band, axis): stacked_state_derivatives[(band, axis)][
                    :, state : state + 1
                ]
                for state in range(5)
                for band in (0, 1)
                for axis in (0, 1)
            }
            atoms: list[torch.Tensor] = []

            def append(advector: torch.Tensor, derivative: torch.Tensor, axis: int) -> None:
                atom = advector[..., axis : axis + 1] * derivative
                if atom.shape[1] == 1:
                    atom = atom.expand(-1, 20, -1, -1, -1)
                atom = atom - atom.mean(dim=(-3, -2), keepdim=True)
                rms = atom.square().mean(dim=(-3, -2, -1), keepdim=True).sqrt()
                atoms.append(atom / rms.clamp_min(1e-12) * scale)

            for first, second in self.band_pairs:
                for axis in (0, 1):
                    append(point_bands[first], point_derivatives[(second, axis)], axis)
            for state in range(5):
                for point_band, state_band in self.band_pairs:
                    for order in ("point_state", "state_point"):
                        for axis in (0, 1):
                            if order == "point_state":
                                append(
                                    point_bands[point_band],
                                    state_derivatives[(state, state_band, axis)],
                                    axis,
                                )
                            else:
                                append(
                                    state_bands[state][state_band],
                                    point_derivatives[(point_band, axis)],
                                    axis,
                                )
            if len(atoms) != self.atom_count:
                raise RuntimeError(f"C442 constructed {len(atoms)} atoms")
            return torch.stack(atoms, dim=2)

    def _synthesize(
        self,
        observed_normalized: torch.Tensor,
        point_normalized: torch.Tensor,
        scale: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        basis = self._point_delay_basis(observed_normalized, point_normalized, scale)
        temporal_coefficients = coefficients[:, :, None].expand(-1, -1, 5, -1).reshape(
            coefficients.shape[0], 20, self.atom_count
        )
        correction = torch.einsum("bta,btahwc->bthwc", temporal_coefficients, basis)
        return self._low_solenoidal(correction)

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
        observed_latent = self._encode_normalized(observed_normalized)
        observed_spectrum = self._select_modes(observed_latent)
        propagated, normality, orthogonality, closure = self._propagate(observed_spectrum)
        point_spectrum = self._select_modes(self._encode_normalized(point_normalized))
        coefficients = self._coefficients(propagated, point_spectrum)
        correction = self._synthesize(
            observed_normalized, point_normalized, scale, coefficients
        )
        prediction_uv = base_prediction[..., :2] + correction
        prediction = torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)

        if return_details:
            details: dict[str, torch.Tensor] = {
                "prediction": prediction,
                "base_prediction": base_prediction,
                "base_half": base_half,
                "coefficients": coefficients,
                "correction": correction,
                "propagated": propagated,
                "point_spectrum": point_spectrum,
                "normality": normality,
                "orthogonality": orthogonality,
                "closure": closure,
            }
            if target is not None:
                target_normalized = (target[..., :2] - mean) / scale
                target_spectrum = self._select_modes(
                    self._encode_normalized(target_normalized)
                )
                latent_loss = (propagated - target_spectrum).abs().square().mean()
                latent_loss = latent_loss / target_spectrum.abs().square().mean().clamp_min(1e-8)
                details["target_spectrum"] = target_spectrum
                details["latent_loss"] = latent_loss
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
        correction = details["correction"]
        closure = details["closure"]
        coefficients = details["coefficients"]
        return {
            "atom_count": self.atom_count,
            "atom_order": list(self.atom_order()),
            "correction_rms": float(correction.float().square().mean().sqrt()),
            "correction_abs_max": float(correction.abs().max()),
            "coefficient_rms": float(coefficients.float().square().mean().sqrt()),
            "coefficient_abs_max": float(coefficients.abs().max()),
            "closure_rms": float(closure.abs().square().mean().sqrt()),
            "normality": float(details["normality"]),
            "orthogonality": float(details["orthogonality"]),
            "all_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    details["prediction"], correction, coefficients,
                    details["propagated"], closure,
                )
            ),
        }
