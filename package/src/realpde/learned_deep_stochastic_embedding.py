"""Learned, window-local Deep Stochastic Embedding operator for C279."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnedDSEConfig:
    in_steps: int = 20
    out_steps: int = 20
    feature_width: int = 48
    delay_length: int = 10
    state_rank: int = 16
    state_dictionary_width: int = 32
    observation_dictionary_width: int = 32
    convolution_width: int = 16
    ridge: float = 1.0e-4

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class FrameEncoder(nn.Module):
    """Time-invariant frame encoder shared by observed and future frames."""

    def __init__(self, width: int, feature_width: int) -> None:
        super().__init__()
        channels = 3 * width // 2
        self.network = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # 32x64 becomes 8x16 after the two strided convolutions.
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(channels * 8, feature_width),
            nn.LayerNorm(feature_width),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)


class FrameDecoder(nn.Module):
    """Time-invariant feature-to-velocity decoder."""

    def __init__(self, width: int, feature_width: int) -> None:
        super().__init__()
        channels = 3 * width // 2
        self.channels = channels
        self.project = nn.Sequential(
            nn.Linear(feature_width, channels * 8 * 16),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.project(features).reshape(-1, self.channels, 8, 16)
        value = F.interpolate(value, size=(32, 64), mode="bilinear", align_corners=False)
        return self.refine(value)


class SharedDelayHead(nn.Module):
    """Apply the same nonlinear delay map to every feature coordinate."""

    def __init__(self, delay_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(delay_length, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("delay sequence must have shape [B,ell,m]")
        return self.network(sequence.transpose(1, 2)).squeeze(-1)


def symmetric_inverse_sqrt(
    matrix: torch.Tensor, ridge: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = (matrix + matrix.T) * 0.5
    if ridge:
        matrix = matrix + float(ridge) * torch.eye(
            matrix.shape[0], dtype=matrix.dtype, device=matrix.device
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(1.0e-8)
    inverse = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    return inverse, eigenvalues


def functional_cca(
    past: torch.Tensor,
    future: torch.Tensor,
    rank: int,
    ridge: float,
) -> dict[str, torch.Tensor]:
    """Regularized CCA used for the frozen global representation fit."""
    if past.ndim != 2 or future.ndim != 2 or past.shape != future.shape:
        raise ValueError("past/future must have the same [N,m] shape")
    p_mean = past.mean(0)
    f_mean = future.mean(0)
    p = past - p_mean
    f = future - f_mean
    denominator = max(1, len(p) - 1)
    cp = p.T @ p / denominator
    cf = f.T @ f / denominator
    cross = f.T @ p / denominator
    inverse_p, eigen_p = symmetric_inverse_sqrt(cp, ridge)
    inverse_f, eigen_f = symmetric_inverse_sqrt(cf, ridge)
    left, singular, right_h = torch.linalg.svd(
        inverse_f @ cross @ inverse_p, full_matrices=False
    )
    take = min(int(rank), singular.numel())
    return {
        "past_mean": p_mean,
        "future_mean": f_mean,
        "past_projection": inverse_p @ right_h.T[:, :take],
        "future_projection": inverse_f @ left[:, :take],
        "singular_values": singular[:take],
        "past_eigenvalues": eigen_p,
        "future_eigenvalues": eigen_f,
    }


def differentiable_cca_score(
    past: torch.Tensor,
    future: torch.Tensor,
    rank: int,
    ridge: float,
) -> torch.Tensor:
    """Mean top-r CCA correlation with stop-gradient canonical directions."""
    if past.ndim != 2 or past.shape != future.shape:
        raise ValueError("past/future must have the same [N,m] shape")
    # Alternating CCA: solve directions exactly for the current batch without
    # differentiating through an ambiguous eigensystem, then differentiate the
    # canonical correlations through both learned representations.
    with torch.no_grad():
        fitted = functional_cca(
            past.detach().double(), future.detach().double(), rank, ridge
        )
        p_mean = fitted["past_mean"].to(past)
        f_mean = fitted["future_mean"].to(future)
        p_projection = fitted["past_projection"].to(past)
        f_projection = fitted["future_projection"].to(future)
    p_score = (past - p_mean) @ p_projection
    f_score = (future - f_mean) @ f_projection
    numerator = (p_score * f_score).mean(0)
    denominator = (
        p_score.square().mean(0).clamp_min(1.0e-8)
        * f_score.square().mean(0).clamp_min(1.0e-8)
    ).sqrt()
    return (numerator / denominator).clamp(-1.0, 1.0).mean()


def ridge_weights(inputs: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    """Return W such that inputs @ W approximates targets."""
    identity = torch.eye(inputs.shape[1], dtype=inputs.dtype, device=inputs.device)
    return torch.linalg.solve(
        inputs.T @ inputs + float(ridge) * identity,
        inputs.T @ targets,
    )


def empirical_covariance(residual: torch.Tensor, ridge: float) -> torch.Tensor:
    centered = residual - residual.mean(0)
    value = centered.T @ centered / max(1, len(centered) - 1)
    return value + float(ridge) * torch.eye(
        value.shape[0], dtype=value.dtype, device=value.device
    )


class LearnedDeepStochasticEmbedding(nn.Module):
    """Causal DSE forecast using only the 20-frame window it answers."""

    semantic_components = (
        "learned_time_invariant_frame_encoder_decoder",
        "shared_nonlinear_delay_head",
        "regularized_past_future_functional_cca",
        "separate_nonlinear_state_observation_dictionaries",
        "closed_form_dense_transfer_observation_operators",
        "empirical_qr_fixed_gain_feature_filter",
        "causal_window_local_twenty_step_rollout",
    )

    def __init__(self, config: LearnedDSEConfig) -> None:
        super().__init__()
        self.config = config
        m = config.feature_width
        r = config.state_rank
        da = config.state_dictionary_width
        db = config.observation_dictionary_width
        self.encoder = FrameEncoder(config.convolution_width, m)
        self.decoder = FrameDecoder(config.convolution_width, m)
        self.delay_head = SharedDelayHead(config.delay_length)
        self.state_dictionary = nn.Sequential(
            nn.Linear(r, da), nn.Tanh(), nn.Linear(da, da)
        )
        self.state_readout = nn.Linear(da, r)
        self.observation_dictionary = nn.Sequential(
            nn.Linear(m, db), nn.Tanh(), nn.Linear(db, db)
        )
        self.observation_readout = nn.Linear(db, m)

        self.register_buffer("feature_mean", torch.zeros(m))
        self.register_buffer("feature_std", torch.ones(m))
        self.register_buffer("past_mean", torch.zeros(m))
        self.register_buffer("canonical_projection", torch.zeros(m, r))
        self.register_buffer("singular_values", torch.ones(r))
        self.register_buffer("state_mean", torch.zeros(da))
        self.register_buffer("observation_mean", torch.zeros(db))
        self.register_buffer("transfer_operator", torch.eye(da))
        self.register_buffer("observation_operator", torch.zeros(db, da))
        self.register_buffer("process_covariance", torch.eye(da) * config.ridge)
        self.register_buffer("observation_covariance", torch.eye(db) * config.ridge)
        self.register_buffer("initial_covariance", torch.eye(da))
        self.register_buffer(
            "kalman_gains",
            torch.zeros(config.in_steps - config.delay_length, da, db),
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5 or tuple(sequence.shape[2:4]) != (32, 64):
            raise ValueError(f"expected [B,T,32,64,C], got {tuple(sequence.shape)}")
        batch, steps, height, width, _ = sequence.shape
        frames = sequence[..., :2].permute(0, 1, 4, 2, 3).reshape(
            batch * steps, 2, height, width
        )
        return self.encoder(frames).reshape(batch, steps, self.config.feature_width)

    def decode_sequence(self, features: torch.Tensor) -> torch.Tensor:
        batch, steps, width = features.shape
        decoded = self.decoder(features.reshape(batch * steps, width))
        return decoded.reshape(batch, steps, 2, 32, 64).permute(
            0, 1, 3, 4, 2
        ).contiguous()

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std.clamp_min(1.0e-6)

    def denormalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.feature_std.clamp_min(1.0e-6) + self.feature_mean

    def block_features(self, normalized: torch.Tensor) -> torch.Tensor:
        ell = self.config.delay_length
        if normalized.shape[1] < ell:
            raise ValueError("sequence shorter than delay length")
        blocks = [
            self.delay_head(normalized[:, end - ell + 1 : end + 1])
            for end in range(ell - 1, normalized.shape[1])
        ]
        return torch.stack(blocks, dim=1)

    def canonical_states(self, blocks: torch.Tensor) -> torch.Tensor:
        return (
            (blocks - self.past_mean) @ self.canonical_projection
        ) * self.singular_values.clamp_min(0.0).sqrt()

    @staticmethod
    def gain_schedule(
        transfer: torch.Tensor,
        observation: torch.Tensor,
        process_covariance: torch.Tensor,
        observation_covariance: torch.Tensor,
        covariance: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        identity = torch.eye(
            transfer.shape[0], dtype=transfer.dtype, device=transfer.device
        )
        gains = []
        value = covariance
        for _ in range(int(steps)):
            prior = transfer @ value @ transfer.T + process_covariance
            innovation = observation @ prior @ observation.T + observation_covariance
            gain = torch.linalg.solve(innovation, observation @ prior).T
            value = (identity - gain @ observation) @ prior
            value = (value + value.T) * 0.5
            gains.append(gain)
        return torch.stack(gains)

    @torch.no_grad()
    def set_representation(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        cca: dict[str, torch.Tensor],
    ) -> None:
        self.feature_mean.copy_(feature_mean.to(self.feature_mean))
        self.feature_std.copy_(feature_std.to(self.feature_std).clamp_min(1.0e-6))
        self.past_mean.copy_(cca["past_mean"].to(self.past_mean))
        projection = cca["past_projection"][:, : self.config.state_rank]
        singular = cca["singular_values"][: self.config.state_rank]
        self.canonical_projection.copy_(projection.to(self.canonical_projection))
        self.singular_values.copy_(singular.to(self.singular_values))

    @torch.no_grad()
    def set_system(
        self,
        *,
        state_mean: torch.Tensor,
        observation_mean: torch.Tensor,
        transfer: torch.Tensor,
        observation: torch.Tensor,
        process_covariance: torch.Tensor,
        observation_covariance: torch.Tensor,
        initial_covariance: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(state_mean.to(self.state_mean))
        self.observation_mean.copy_(observation_mean.to(self.observation_mean))
        self.transfer_operator.copy_(transfer.to(self.transfer_operator))
        self.observation_operator.copy_(observation.to(self.observation_operator))
        self.process_covariance.copy_(process_covariance.to(self.process_covariance))
        self.observation_covariance.copy_(
            observation_covariance.to(self.observation_covariance)
        )
        self.initial_covariance.copy_(initial_covariance.to(self.initial_covariance))
        gains = self.gain_schedule(
            self.transfer_operator,
            self.observation_operator,
            self.process_covariance,
            self.observation_covariance,
            self.initial_covariance,
            self.config.in_steps - self.config.delay_length,
        )
        self.kalman_gains.copy_(gains)

    def forecast_from_features(self, observed_features: torch.Tensor) -> torch.Tensor:
        if observed_features.shape[1] != self.config.in_steps:
            raise ValueError("forecast requires exactly 20 observed features")
        normalized = self.normalize_features(observed_features)
        blocks = self.block_features(normalized)
        canonical = self.canonical_states(blocks)
        state = self.state_dictionary(canonical[:, 0]) - self.state_mean
        for offset, time_index in enumerate(
            range(self.config.delay_length, self.config.in_steps)
        ):
            prior = state @ self.transfer_operator.T
            observation = (
                self.observation_dictionary(normalized[:, time_index])
                - self.observation_mean
            )
            innovation = observation - prior @ self.observation_operator.T
            state = prior + innovation @ self.kalman_gains[offset].T

        future_features = []
        for _ in range(self.config.out_steps):
            state = state @ self.transfer_operator.T
            observation = state @ self.observation_operator.T + self.observation_mean
            normalized_feature = self.observation_readout(observation)
            future_features.append(self.denormalize_features(normalized_feature))
        future = torch.stack(future_features, dim=1)
        return self.decode_sequence(future)

    def forward(self, input_array: torch.Tensor) -> torch.Tensor:
        if input_array.ndim != 5 or input_array.shape[1] != self.config.in_steps:
            raise ValueError(
                f"expected [B,{self.config.in_steps},H,W,C], got {tuple(input_array.shape)}"
            )
        return self.forecast_from_features(self.encode_sequence(input_array))

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        # Fixed singleton execution removes CUDA batch-shape roundoff while
        # preserving a strictly window-local, deterministic graph.
        return torch.cat(
            [self.forward(input_array[index : index + 1]) for index in range(len(input_array))],
            dim=0,
        )

