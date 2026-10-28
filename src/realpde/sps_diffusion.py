from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import OperatorBlock
from .sps_ordinal import SPSOrdinalConfig, SPSOrdinalUncertaintyOperator


@dataclass(frozen=True)
class SPSFrequencyFlowConfig:
    """Probability flow over the complete temporal spectrum of absolute residuals."""

    ordinal: dict[str, Any]
    width: int = 64
    depth: int = 5
    spectral_rank: int = 16
    modes_y: int = 12
    modes_x: int = 24
    expansion: int = 2
    dropout: float = 0.02
    time_embedding_channels: int = 16
    frequency_bins: int = 11
    log_epsilon: float = 0.05
    dc_scale: float = 4.0
    target_clip: float = 8.0
    sample_steps: int = 4
    ensemble_size: int = 8
    max_ensemble_size: int = 8
    sample_seed: int = 20260810

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SPSFrequencyFlowConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in dict(values).items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SPSFrequencyResidualFlow(nn.Module):
    """Generate normalized absolute errors in a bijective temporal-rFFT basis."""

    def __init__(self, config: SPSFrequencyFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.ordinal = SPSOrdinalUncertaintyOperator(
            SPSOrdinalConfig.from_dict(config.ordinal)
        )
        self.ordinal.requires_grad_(False)
        self.ordinal.eval()
        self.out_steps = self.ordinal.out_steps
        expected_bins = self.out_steps // 2 + 1
        if config.frequency_bins != expected_bins:
            raise ValueError(
                f"frequency_bins must preserve the full rFFT: {expected_bins}"
            )
        if config.sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if not 1 <= config.ensemble_size <= config.max_ensemble_size:
            raise ValueError("ensemble_size must fit the deterministic noise bank")
        self.latent_channels = config.frequency_bins * 2 * 2
        condition_channels = self.ordinal.config.width
        self.lift = nn.Conv2d(
            self.latent_channels + condition_channels + config.time_embedding_channels,
            config.width,
            1,
        )
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.velocity_head = nn.Sequential(
            nn.GroupNorm(groups, config.width),
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, self.latent_channels, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        channel_scale = torch.ones(self.latent_channels)
        channel_scale[: 2 * 2] = float(config.dc_scale)
        self.register_buffer("channel_scale", channel_scale, persistent=True)
        generator = torch.Generator(device="cpu").manual_seed(config.sample_seed)
        half = (config.max_ensemble_size + 1) // 2
        noise = torch.randn(half, self.latent_channels, 32, 64, generator=generator)
        antithetic = torch.stack((noise, -noise), dim=1).flatten(0, 1)
        self.register_buffer(
            "noise_bank", antithetic[: config.max_ensemble_size], persistent=True
        )

    def train(self, mode: bool = True) -> SPSFrequencyResidualFlow:
        super().train(mode)
        self.ordinal.eval()
        return self

    def _time_embedding(self, time_value: torch.Tensor) -> torch.Tensor:
        half = self.config.time_embedding_channels // 2
        frequency = 2.0 ** torch.arange(
            half, device=time_value.device, dtype=time_value.dtype
        )
        phase = math.pi * time_value[:, None] * frequency[None]
        embedded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        if embedded.shape[1] < self.config.time_embedding_channels:
            embedded = F.pad(
                embedded, (0, self.config.time_embedding_channels - embedded.shape[1])
            )
        return embedded[:, :, None, None].expand(-1, -1, 32, 64)

    @torch.no_grad()
    def condition(self, observed: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.ordinal.features(observed, point)
        return hidden

    def velocity(
        self, latent: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.lift(
            torch.cat((latent, condition, self._time_embedding(time_value)), dim=1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden)

    def encode_error(self, normalized_error: torch.Tensor) -> torch.Tensor:
        transformed = torch.log1p(
            normalized_error.clamp_min(0.0) / float(self.config.log_epsilon)
        )
        spectrum = torch.fft.rfft(transformed.float(), dim=1, norm="ortho")
        packed = torch.view_as_real(spectrum)
        latent = packed.permute(0, 1, 4, 5, 2, 3).reshape(
            transformed.shape[0], self.latent_channels, 32, 64
        )
        scale = self.channel_scale.view(1, -1, 1, 1)
        return (latent / scale).clamp(
            -float(self.config.target_clip), float(self.config.target_clip)
        )

    def decode_error(self, latent: torch.Tensor) -> torch.Tensor:
        scale = self.channel_scale.view(1, -1, 1, 1)
        packed = (latent * scale).reshape(
            latent.shape[0], self.config.frequency_bins, 2, 2, 32, 64
        )
        packed = packed.permute(0, 1, 4, 5, 2, 3).contiguous()
        spectrum = torch.view_as_complex(packed)
        transformed = torch.fft.irfft(
            spectrum, n=self.out_steps, dim=1, norm="ortho"
        ).clamp(0.0, 12.0)
        return torch.expm1(transformed) * float(self.config.log_epsilon)

    def flow_matching_loss(
        self,
        observed: torch.Tensor,
        point: torch.Tensor,
        target: torch.Tensor,
        values: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        uv = observed[..., :2].float()
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_error = (point[..., :2] - target[..., :2]).abs() / scale
        target_latent = self.encode_error(normalized_error)
        condition = self.condition(observed, point)
        noise = torch.randn_like(target_latent)
        time_value = torch.rand(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        shaped_time = time_value[:, None, None, None]
        interpolated = (1.0 - shaped_time) * noise + shaped_time * target_latent
        velocity = self.velocity(interpolated, time_value, condition)
        velocity_target = target_latent - noise
        endpoint = interpolated + (1.0 - shaped_time) * velocity
        decoded = self.decode_error(endpoint)
        flow_loss = F.mse_loss(velocity, velocity_target)
        endpoint_loss = F.smooth_l1_loss(endpoint, target_latent, beta=0.25)
        reconstruction = F.smooth_l1_loss(
            torch.log1p(decoded / float(self.config.log_epsilon)),
            torch.log1p(normalized_error / float(self.config.log_epsilon)),
            beta=0.25,
        )
        loss = (
            float(values.get("flow_weight", 1.0)) * flow_loss
            + float(values.get("endpoint_weight", 0.5)) * endpoint_loss
            + float(values.get("reconstruction_weight", 0.5)) * reconstruction
        )
        return loss, {
            "loss": float(loss.detach()),
            "flow": float(flow_loss.detach()),
            "endpoint": float(endpoint_loss.detach()),
            "reconstruction": float(reconstruction.detach()),
        }

    def _integrate(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        steps = int(self.config.sample_steps)
        result = latent
        for index in range(steps):
            time_value = result.new_full((result.shape[0],), index / steps)
            result = result + self.velocity(result, time_value, condition) / steps
        return result

    @torch.inference_mode()
    def sample_normalized_errors(
        self,
        observed: torch.Tensor,
        point: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is None:
            condition = self.condition(observed, point)
        ensemble = int(self.config.ensemble_size)
        if not 1 <= ensemble <= int(self.config.max_ensemble_size):
            raise ValueError("ensemble_size exceeds deterministic noise bank")
        batch = observed.shape[0]
        repeated_condition = condition[:, None].expand(
            -1, ensemble, -1, -1, -1
        ).reshape(batch * ensemble, condition.shape[1], 32, 64)
        initial = self.noise_bank[:ensemble][None].expand(
            batch, -1, -1, -1, -1
        ).reshape(batch * ensemble, self.latent_channels, 32, 64)
        sampled = self._integrate(initial.to(condition.dtype), repeated_condition)
        decoded = self.decode_error(sampled)
        return decoded.reshape(batch, ensemble, self.out_steps, 32, 64, 2)

    @torch.inference_mode()
    def ordinal_width(
        self, observed: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        logits, scale = self.ordinal.cdf_logits(observed, point)
        width_uv = self.ordinal.decision_width(logits, scale)
        return torch.cat((width_uv, torch.zeros_like(width_uv[..., :1])), dim=-1)


def initialize_frequency_flow(
    ordinal_checkpoint: str | Path,
    values: dict[str, Any],
    device: torch.device,
) -> SPSFrequencyResidualFlow:
    checkpoint = torch.load(ordinal_checkpoint, map_location=device)
    if checkpoint.get("model_type") != "sps_ordinal_uncertainty":
        raise ValueError(f"not an SPS ordinal checkpoint: {ordinal_checkpoint}")
    model_values = dict(values)
    model_values["ordinal"] = checkpoint["model_config"]
    model = SPSFrequencyResidualFlow(
        SPSFrequencyFlowConfig.from_dict(model_values)
    ).to(device)
    model.ordinal.load_state_dict(checkpoint["state_dict"], strict=True)
    model.ordinal.requires_grad_(False)
    model.ordinal.eval()
    return model


def save_frequency_flow_checkpoint(
    path: str | Path,
    model: SPSFrequencyResidualFlow,
    config: dict[str, Any],
    *,
    step: int,
    metrics: dict[str, Any],
    deployment: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "sps_frequency_residual_flow",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": "fno_tangent_frequency_flow",
            "step": int(step),
            "metrics": metrics,
            "deployment": deployment,
            "source_config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        },
        output,
    )
    return output


def load_frequency_flow_checkpoint(
    path: str | Path, device: torch.device
) -> SPSFrequencyResidualFlow:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "sps_frequency_residual_flow":
        raise ValueError(f"not an SPS frequency-flow checkpoint: {path}")
    model = SPSFrequencyResidualFlow(
        SPSFrequencyFlowConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
