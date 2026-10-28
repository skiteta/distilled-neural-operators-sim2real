"""Faithful Deep Stochastic Embedding primitives for the C278 feasibility gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DSEConfig:
    in_steps: int = 20
    out_steps: int = 20
    feature_width: int = 48
    delay_length: int = 10
    state_rank: int = 16
    state_dictionary_width: int = 32
    observation_dictionary_width: int = 32
    convolution_width: int = 16

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class FrameEncoder(nn.Module):
    """One time-invariant encoder shared by every observed frame."""

    def __init__(self, width: int, feature_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, 3 * width // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear((3 * width // 2) * 8, feature_width),
            nn.LayerNorm(feature_width),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)


class FrameDecoder(nn.Module):
    """One time-invariant decoder shared by every predicted frame."""

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
    """Shared nonlinear h from each feature's delay vector to one block feature."""

    def __init__(self, delay_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(delay_length, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # [batch, delay, feature] -> shared h over every feature dimension.
        value = sequence.transpose(1, 2)
        return self.network(value).squeeze(-1)


class FaithfulDSEPrototype(nn.Module):
    """Window-local DSE graph with CCA state, dictionaries and feature filtering.

    C278 initializes this graph randomly only to exercise semantic completeness,
    independence and runtime. A later fit contract would estimate all learned
    components and closed-form buffers from released training data.
    """

    semantic_components = (
        "time_invariant_learnable_encoder_decoder",
        "shared_nonlinear_delay_head",
        "regularized_past_future_functional_cca",
        "separate_state_observation_dictionaries",
        "closed_form_transfer_observation_operators",
        "empirical_qr_feature_kalman_filter",
        "causal_window_local_future_rollout",
    )

    def __init__(self, config: DSEConfig) -> None:
        super().__init__()
        self.config = config
        m = config.feature_width
        r = config.state_rank
        d_a = config.state_dictionary_width
        d_b = config.observation_dictionary_width
        self.encoder = FrameEncoder(config.convolution_width, m)
        self.decoder = FrameDecoder(config.convolution_width, m)
        self.delay_head = SharedDelayHead(config.delay_length)
        self.state_dictionary = nn.Sequential(
            nn.Linear(r, d_a),
            nn.Tanh(),
            nn.Linear(d_a, d_a),
        )
        self.observation_dictionary = nn.Sequential(
            nn.Linear(m, d_b),
            nn.Tanh(),
            nn.Linear(d_b, d_b),
        )
        self.observation_readout = nn.Linear(d_b, m)

        generator = torch.Generator().manual_seed(20260820)
        basis = torch.randn(m, r, generator=generator)
        basis = torch.linalg.qr(basis, mode="reduced").Q
        q_a = torch.linalg.qr(torch.randn(d_a, d_a, generator=generator)).Q
        rates = torch.linspace(0.82, 0.97, d_a)
        transfer = q_a @ torch.diag(rates) @ q_a.T
        observation = torch.randn(d_b, d_a, generator=generator) / (d_a ** 0.5)
        self.register_buffer("feature_mean", torch.zeros(m))
        self.register_buffer("feature_std", torch.ones(m))
        self.register_buffer("past_mean", torch.zeros(m))
        self.register_buffer("canonical_projection", basis)
        self.register_buffer("transfer_operator", transfer)
        self.register_buffer("observation_operator", observation)
        self.register_buffer("process_covariance", torch.eye(d_a) * 0.01)
        self.register_buffer("observation_covariance", torch.eye(d_b) * 0.05)
        self.register_buffer("initial_covariance", torch.eye(d_a) * 0.10)
        self.register_buffer(
            "kalman_gains",
            self._gain_schedule(
                transfer,
                observation,
                torch.eye(d_a) * 0.01,
                torch.eye(d_b) * 0.05,
                torch.eye(d_a) * 0.10,
                config.in_steps - config.delay_length,
            ),
        )

    @staticmethod
    def _gain_schedule(
        transfer: torch.Tensor,
        observation: torch.Tensor,
        process_covariance: torch.Tensor,
        observation_covariance: torch.Tensor,
        covariance: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        identity = torch.eye(transfer.shape[0], dtype=transfer.dtype)
        gains = []
        value = covariance
        for _ in range(steps):
            prior = transfer @ value @ transfer.T + process_covariance
            innovation = observation @ prior @ observation.T + observation_covariance
            gain = torch.linalg.solve(innovation, observation @ prior).T
            posterior = (identity - gain @ observation) @ prior
            posterior = (posterior + posterior.T) * 0.5
            gains.append(gain)
            value = posterior
        return torch.stack(gains)

    def forward(self, input_array: torch.Tensor) -> torch.Tensor:
        if input_array.ndim != 5 or input_array.shape[1] != self.config.in_steps:
            raise ValueError(f"expected [B,{self.config.in_steps},H,W,C], got {input_array.shape}")
        batch, steps, height, width, _channels = input_array.shape
        frames = input_array[..., :2].permute(0, 1, 4, 2, 3).reshape(
            batch * steps, 2, height, width
        )
        features = self.encoder(frames).reshape(batch, steps, self.config.feature_width)
        features = (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)

        block = self.delay_head(features[:, : self.config.delay_length])
        canonical = (block - self.past_mean) @ self.canonical_projection
        state = self.state_dictionary(canonical)
        for offset, time_index in enumerate(
            range(self.config.delay_length, self.config.in_steps)
        ):
            prior = state @ self.transfer_operator.T
            observation = self.observation_dictionary(features[:, time_index])
            innovation = observation - prior @ self.observation_operator.T
            state = prior + innovation @ self.kalman_gains[offset].T

        future_features = []
        for _ in range(self.config.out_steps):
            state = state @ self.transfer_operator.T
            observation = state @ self.observation_operator.T
            future_features.append(self.observation_readout(observation))
        future = torch.stack(future_features, dim=1)
        decoded = self.decoder(future.reshape(batch * self.config.out_steps, -1))
        return decoded.reshape(batch, self.config.out_steps, 2, 32, 64).permute(
            0, 1, 3, 4, 2
        ).contiguous()

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)


def symmetric_inverse_sqrt(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    if bool((eigenvalues <= 0).any()):
        raise RuntimeError("matrix is not positive definite")
    inverse = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    return inverse, eigenvalues


def functional_cca(
    past: torch.Tensor,
    future: torch.Tensor,
    rank: int,
    ridge: float,
) -> dict[str, torch.Tensor]:
    if past.ndim != 2 or future.shape != past.shape:
        raise ValueError("past/future blocks must have the same [N,m] shape")
    past_mean = past.mean(0)
    future_mean = future.mean(0)
    p = past - past_mean
    f = future - future_mean
    normalizer = max(1, p.shape[0] - 1)
    identity = torch.eye(p.shape[1], dtype=p.dtype, device=p.device)
    c_p = p.T @ p / normalizer + ridge * identity
    c_f = f.T @ f / normalizer + ridge * identity
    h = f.T @ p / normalizer
    inverse_p, eigen_p = symmetric_inverse_sqrt(c_p)
    inverse_f, eigen_f = symmetric_inverse_sqrt(c_f)
    whitened = inverse_f @ h @ inverse_p
    left, singular, right_h = torch.linalg.svd(whitened, full_matrices=False)
    take = min(rank, singular.numel())
    projection_p = inverse_p @ right_h.T[:, :take]
    projection_f = inverse_f @ left[:, :take]
    return {
        "past_mean": past_mean,
        "future_mean": future_mean,
        "past_covariance": c_p,
        "future_covariance": c_f,
        "past_eigenvalues": eigen_p,
        "future_eigenvalues": eigen_f,
        "past_projection": projection_p,
        "future_projection": projection_f,
        "singular_values": singular[:take],
    }


def ridge_solve(inputs: torch.Tensor, targets: torch.Tensor, ridge: float) -> tuple[torch.Tensor, float]:
    gram = inputs.T @ inputs
    regularized = gram + ridge * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    weights = torch.linalg.solve(regularized, inputs.T @ targets)
    condition = float(torch.linalg.cond(regularized))
    return weights, condition
