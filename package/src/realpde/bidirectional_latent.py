from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import turbulent_kinetic_energy


@dataclass(frozen=True)
class BidirectionalLatentConfig:
    """Compact past/future state model with training-only future information."""

    in_steps: int = 20
    width: int = 32
    latent_channels: int = 32
    encoder_depth: int = 2
    dynamics_depth: int = 3
    maximum_log_tke_ratio: float = 0.7
    minimum_log_scale: float = -4.0
    maximum_log_scale: float = 1.0
    trajectory_frequency_cutoff: int | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BidirectionalLatentConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.channel = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(field)
        update = self.channel(F.gelu(self.depthwise(normalized)))
        return field + self.scale * update


class BlockStateEncoder(nn.Module):
    """Encode a 20-frame velocity block into an 8x16 hidden-state map."""

    def __init__(self, config: BidirectionalLatentConfig) -> None:
        super().__init__()
        # 20*(u,v), nine local physical summaries, and two coordinates.
        input_channels = 2 * config.in_steps + 9 + 2
        self.lift = nn.Conv2d(input_channels, config.width, 3, padding=1)
        self.full_blocks = nn.ModuleList(
            [ResidualConvBlock(config.width) for _ in range(config.encoder_depth)]
        )
        self.down1 = nn.Conv2d(config.width, config.width, 3, stride=2, padding=1)
        self.half_block = ResidualConvBlock(config.width)
        self.down2 = nn.Conv2d(
            config.width, config.latent_channels, 3, stride=2, padding=1
        )
        self.latent_block = ResidualConvBlock(config.latent_channels)

    @staticmethod
    def _features(flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 20 or flow.shape[-1] < 2:
            raise ValueError(f"expected [B,20,H,W,C>=2], got {tuple(flow.shape)}")
        uv = flow[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (uv - mean) / std
        batch, _, height, width, _ = normalized.shape
        sequence = normalized.permute(0, 1, 4, 2, 3).reshape(
            batch, 40, height, width
        )
        temporal_mean = normalized.mean(dim=1)
        fluctuation = normalized - temporal_mean[:, None]
        temporal_std = fluctuation.square().mean(dim=1).clamp_min(1e-8).sqrt()
        local_tke = 0.5 * fluctuation.square().sum(dim=-1).mean(dim=1, keepdim=False)
        local_tke = torch.log1p(local_tke)[..., None]
        recent_trend = (normalized[:, -1] - normalized[:, -5]) / 4.0
        final_fluctuation = normalized[:, -1] - temporal_mean
        summaries = torch.cat(
            (temporal_mean, temporal_std, local_tke, recent_trend, final_fluctuation),
            dim=-1,
        ).permute(0, 3, 1, 2)
        yy = torch.linspace(-1.0, 1.0, height, device=flow.device)
        xx = torch.linspace(-1.0, 1.0, width, device=flow.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x), dim=0).expand(batch, -1, -1, -1)
        return torch.cat((sequence, summaries, coordinates), dim=1)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.lift(self._features(flow)))
        for block in self.full_blocks:
            hidden = block(hidden)
        hidden = self.half_block(F.gelu(self.down1(hidden)))
        return self.latent_block(F.gelu(self.down2(hidden)))


class LatentTransition(nn.Module):
    """Move a hidden state forward or backward by one 20-frame block."""

    def __init__(self, channels: int, depth: int, *, probabilistic: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualConvBlock(channels) for _ in range(depth)])
        self.mean_head = nn.Conv2d(channels, channels, 1)
        self.log_scale_head = nn.Conv2d(channels, channels, 1) if probabilistic else None
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        if self.log_scale_head is not None:
            nn.init.zeros_(self.log_scale_head.weight)
            nn.init.constant_(self.log_scale_head.bias, -2.0)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = state
        for block in self.blocks:
            hidden = block(hidden)
        mean = state + self.mean_head(hidden)
        log_scale = self.log_scale_head(hidden) if self.log_scale_head is not None else None
        return mean, log_scale


class SpatialStateDecoder(nn.Module):
    def __init__(self, channels: int, width: int, output_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.block1 = ResidualConvBlock(width)
        self.block2 = ResidualConvBlock(width)
        self.output = nn.Conv2d(width, output_channels, 3, padding=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input(state))
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.block1(hidden)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.output(self.block2(hidden))


class FrequencyStateDecoder(nn.Module):
    """Decode compact temporal Fourier amplitude and phase maps."""

    def __init__(
        self, channels: int, width: int, *, steps: int, cutoff: int
    ) -> None:
        super().__init__()
        if cutoff < 0 or cutoff >= steps // 2:
            raise ValueError(
                f"frequency cutoff must be in [0,{steps // 2 - 1}], got {cutoff}"
            )
        self.steps = int(steps)
        self.cutoff = int(cutoff)
        # Per velocity component: one real DC value, followed by real and
        # imaginary values for modes 1..cutoff. This is 22 channels at the
        # selected Track-1 cutoff, versus 40 direct frame channels.
        packed_per_component = 1 + 2 * self.cutoff
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.block1 = ResidualConvBlock(width)
        self.block2 = ResidualConvBlock(width)
        self.output = nn.Conv2d(width, 2 * packed_per_component, 3, padding=1)
        time = torch.arange(self.steps, dtype=torch.float32)[:, None]
        mode = torch.arange(1, self.cutoff + 1, dtype=torch.float32)[None]
        phase = 2.0 * math.pi * time * mode / self.steps
        basis = torch.cat(
            (
                torch.full((self.steps, 1), 1.0 / math.sqrt(self.steps)),
                (2.0 / math.sqrt(self.steps)) * torch.cos(phase),
                (-2.0 / math.sqrt(self.steps)) * torch.sin(phase),
            ),
            dim=1,
        )
        self.register_buffer("inverse_basis", basis, persistent=False)

    def packed_coefficients(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input(state))
        hidden = F.interpolate(
            hidden, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        hidden = self.block1(hidden)
        hidden = F.interpolate(
            hidden, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        packed = self.output(self.block2(hidden))
        batch, _, height, width = packed.shape
        return packed.reshape(batch, 2, 1 + 2 * self.cutoff, height, width)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        packed = self.packed_coefficients(state)
        # For a short, fixed band, this real inverse-DFT matrix is mathematically
        # identical to irfft but avoids a complex FFT kernel at every inference.
        real_dtype = torch.float64 if packed.dtype == torch.float64 else torch.float32
        basis = self.inverse_basis.to(real_dtype)
        return torch.einsum("tk,bckhw->bthwc", basis, packed.to(real_dtype))


class BidirectionalLatentState(nn.Module):
    """Learn a future state with a training-only future teacher.

    At inference ``predict_log_tke_ratio`` uses only ``observed``. During
    training the same encoder reads the true future, the backward transition
    must recover the past state, and both directions are constrained to form a
    cycle. The future branch is therefore privileged supervision, not a test
    input.
    """

    summary_channels = 9

    def __init__(self, config: BidirectionalLatentConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BlockStateEncoder(config)
        self.teacher_fusion = nn.Sequential(
            nn.Conv2d(2 * config.latent_channels, config.latent_channels, 1),
            nn.GroupNorm(8 if config.latent_channels % 8 == 0 else 1, config.latent_channels),
            nn.GELU(),
        )
        self.forward_transition = LatentTransition(
            config.latent_channels, config.dynamics_depth, probabilistic=True
        )
        self.backward_transition = LatentTransition(
            config.latent_channels, config.dynamics_depth, probabilistic=False
        )
        self.ratio_decoder = SpatialStateDecoder(
            config.latent_channels, config.width, 1
        )
        self.summary_decoder = SpatialStateDecoder(
            config.latent_channels, config.width, self.summary_channels
        )
        if config.trajectory_frequency_cutoff is None:
            self.trajectory_decoder: SpatialStateDecoder | FrequencyStateDecoder = (
                SpatialStateDecoder(
                    config.latent_channels, config.width, 2 * config.in_steps
                )
            )
        else:
            self.trajectory_decoder = FrequencyStateDecoder(
                config.latent_channels,
                config.width,
                steps=config.in_steps,
                cutoff=int(config.trajectory_frequency_cutoff),
            )
        nn.init.zeros_(self.ratio_decoder.output.weight)
        nn.init.zeros_(self.ratio_decoder.output.bias)
        nn.init.zeros_(self.trajectory_decoder.output.weight)
        nn.init.zeros_(self.trajectory_decoder.output.bias)

    @staticmethod
    def block_summary(flow: torch.Tensor) -> torch.Tensor:
        uv = flow[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = (uv - mean) / std
        temporal_mean = normalized.mean(dim=1)
        fluctuation = normalized - temporal_mean[:, None]
        temporal_std = fluctuation.square().mean(dim=1).clamp_min(1e-8).sqrt()
        local_tke = torch.log1p(
            0.5 * fluctuation.square().sum(dim=-1).mean(dim=1)
        )[..., None]
        recent_trend = (normalized[:, -1] - normalized[:, -5]) / 4.0
        final_fluctuation = normalized[:, -1] - temporal_mean
        return torch.cat(
            (temporal_mean, temporal_std, local_tke, recent_trend, final_fluctuation),
            dim=-1,
        ).permute(0, 3, 1, 2)

    @staticmethod
    def oracle_log_tke_ratio(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_tke = turbulent_kinetic_energy(base).clamp_min(1e-8)
        true_tke = turbulent_kinetic_energy(target).clamp_min(1e-8)
        return torch.log(true_tke / base_tke)

    @staticmethod
    def apply_log_tke_ratio(
        prediction: torch.Tensor, log_ratio: torch.Tensor, scale: float
    ) -> torch.Tensor:
        uv = prediction[..., :2].float()
        mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - mean
        factor = torch.exp(0.5 * float(scale) * log_ratio)[..., None]
        corrected = mean + fluctuation * factor[:, None]
        return torch.cat((corrected, torch.zeros_like(corrected[..., :1])), dim=-1)

    def bounded_ratio(self, state: torch.Tensor) -> torch.Tensor:
        maximum = float(self.config.maximum_log_tke_ratio)
        raw = self.ratio_decoder(state)[:, 0]
        return maximum * torch.tanh(raw / max(maximum, 1e-6))

    @staticmethod
    def observation_statistics(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uv = observed[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        return mean, std

    def decoded_normalized_trajectory(self, state: torch.Tensor) -> torch.Tensor:
        decoded = self.trajectory_decoder(state)
        if isinstance(self.trajectory_decoder, FrequencyStateDecoder):
            return decoded
        batch, _, height, width = decoded.shape
        return decoded.reshape(batch, self.config.in_steps, 2, height, width).permute(
            0, 1, 3, 4, 2
        )

    def project_normalized_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Project a target onto the same temporal subspace as the decoder."""
        cutoff = self.config.trajectory_frequency_cutoff
        if cutoff is None:
            return trajectory
        spectrum = torch.fft.rfft(trajectory.float(), dim=1, norm="ortho")
        spectrum[:, int(cutoff) + 1 :] = 0
        return torch.fft.irfft(
            spectrum, n=self.config.in_steps, dim=1, norm="ortho"
        )

    def predict_correction(self, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        future_mean, future_log_scale = self.predict_state(observed)
        _, std = self.observation_statistics(observed)
        correction = self.decoded_normalized_trajectory(future_mean) * std
        return correction, future_log_scale

    @staticmethod
    def tke_preserving_point(
        anchor: torch.Tensor, correction: torch.Tensor, scale: float | torch.Tensor
    ) -> torch.Tensor:
        """Apply a phase/mean correction while preserving anchor local TKE exactly."""
        anchor_uv = anchor[..., :2].float()
        correction_uv = correction.float()
        anchor_mean = anchor_uv.mean(dim=1, keepdim=True)
        correction_mean = correction_uv.mean(dim=1, keepdim=True)
        anchor_fluctuation = anchor_uv - anchor_mean
        correction_fluctuation = correction_uv - correction_mean
        radial = (anchor_fluctuation * correction_fluctuation).sum(
            dim=(1, 4), keepdim=True
        )
        anchor_energy = anchor_fluctuation.square().sum(dim=(1, 4), keepdim=True)
        tangent = correction_fluctuation - anchor_fluctuation * (
            radial / anchor_energy.clamp_min(1e-12)
        )
        scale_value = torch.as_tensor(scale, device=anchor.device, dtype=anchor_uv.dtype)
        while scale_value.ndim < anchor_uv.ndim:
            scale_value = scale_value.unsqueeze(-1)
        candidate = anchor_fluctuation + scale_value * tangent
        candidate_energy = candidate.square().sum(dim=(1, 4), keepdim=True)
        retraction = (anchor_energy / candidate_energy.clamp_min(1e-12)).sqrt()
        retraction = torch.where(
            anchor_energy > 1e-12, retraction, torch.zeros_like(retraction)
        )
        point_mean = anchor_mean + scale_value * correction_mean
        velocity = point_mean + candidate * retraction
        return torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)

    def predict_state(
        self, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_state = self.encoder(observed)
        future_mean, future_log_scale = self.forward_transition(past_state)
        if future_log_scale is None:
            raise RuntimeError("forward transition must be probabilistic")
        future_log_scale = future_log_scale.clamp(
            float(self.config.minimum_log_scale),
            float(self.config.maximum_log_scale),
        )
        return future_mean, future_log_scale

    def predict_log_tke_ratio(
        self, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        future_mean, future_log_scale = self.predict_state(observed)
        return self.bounded_ratio(future_mean), future_log_scale

    def training_outputs(
        self,
        observed: torch.Tensor,
        target: torch.Tensor,
        base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        past_state = self.encoder(observed)
        encoded_future_state = self.encoder(target)
        teacher_future_state = self.teacher_fusion(
            torch.cat((past_state, encoded_future_state), dim=1)
        )
        predicted_future_state, future_log_scale = self.forward_transition(past_state)
        if future_log_scale is None:
            raise RuntimeError("forward transition must be probabilistic")
        predicted_past_state, _ = self.backward_transition(teacher_future_state)
        cycled_past_state, _ = self.backward_transition(predicted_future_state)
        cycled_future_state, _ = self.forward_transition(predicted_past_state)
        return {
            "past_state": past_state,
            "teacher_future_state": teacher_future_state,
            "predicted_future_state": predicted_future_state,
            "predicted_past_state": predicted_past_state,
            "cycled_past_state": cycled_past_state,
            "cycled_future_state": cycled_future_state,
            "future_log_scale": future_log_scale.clamp(
                float(self.config.minimum_log_scale),
                float(self.config.maximum_log_scale),
            ),
            "teacher_ratio": self.bounded_ratio(teacher_future_state),
            "student_ratio": self.bounded_ratio(predicted_future_state),
            "oracle_ratio": self.oracle_log_tke_ratio(base, target).clamp(
                -float(self.config.maximum_log_tke_ratio),
                float(self.config.maximum_log_tke_ratio),
            ),
            "teacher_future_summary": self.summary_decoder(teacher_future_state),
            "student_future_summary": self.summary_decoder(predicted_future_state),
            "predicted_past_summary": self.summary_decoder(predicted_past_state),
            "future_summary": self.block_summary(target),
            "past_summary": self.block_summary(observed),
            "teacher_trajectory": self.decoded_normalized_trajectory(
                teacher_future_state
            ),
            "student_trajectory": self.decoded_normalized_trajectory(
                predicted_future_state
            ),
            "oracle_trajectory": self.project_normalized_trajectory(
                (target[..., :2].float() - base[..., :2].float())
                / self.observation_statistics(observed)[1]
            ),
        }
