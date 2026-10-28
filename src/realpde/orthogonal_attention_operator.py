from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from realpde.model import HybridFlowOperator, ModelConfig


class OrthogonalIntegralAttention2d(nn.Module):
    """Softmax-free orthogonal integral operator over a 2-D latent field.

    This clean implementation follows the learned-feature whitening and
    ``Phi diag(mu) Phi^T f`` update in Xiao et al. (ICML 2024):
    https://proceedings.mlr.press/v235/xiao24c.html
    and the MIT-licensed reference implementation:
    https://github.com/SJTU-DENG-Lab/Orthogonal-Neural-operator

    The running covariance is a training statistic, analogous to a learned
    normalization buffer.  Evaluation never updates it, so no information is
    shared between evaluation windows or calls.
    """

    def __init__(
        self,
        channels: int,
        psi_dim: int = 16,
        momentum: float = 0.9,
        jitter: float = 1e-4,
    ) -> None:
        super().__init__()
        if channels <= 0 or psi_dim <= 0:
            raise ValueError("channels and psi_dim must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if jitter <= 0.0:
            raise ValueError("jitter must be positive")
        self.channels = int(channels)
        self.psi_dim = int(psi_dim)
        self.momentum = float(momentum)
        self.jitter = float(jitter)
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.projection = nn.Linear(channels, psi_dim)
        self.log_eigenvalues = nn.Parameter(torch.zeros(psi_dim))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("feature_cov", torch.eye(psi_dim))

    def _tokens(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},H,W], got {tuple(features.shape)}"
            )
        return self.norm(features).flatten(2).transpose(1, 2)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(self._tokens(features))

    def covariance(self, raw_basis: torch.Tensor) -> torch.Tensor:
        if raw_basis.ndim != 3 or raw_basis.shape[-1] != self.psi_dim:
            raise ValueError("raw_basis must have shape [B,N,psi_dim]")
        count = raw_basis.shape[0] * raw_basis.shape[1]
        return torch.einsum("bnp,bnq->pq", raw_basis, raw_basis) / count

    def whiten(
        self, raw_basis: torch.Tensor, covariance: torch.Tensor
    ) -> torch.Tensor:
        identity = torch.eye(
            self.psi_dim, device=covariance.device, dtype=covariance.dtype
        )
        factor = torch.linalg.cholesky(covariance + self.jitter * identity)
        # solve L Y^T = Phi^T, hence Y = Phi L^{-T}
        return torch.linalg.solve_triangular(
            factor,
            raw_basis.transpose(-1, -2),
            upper=False,
        ).transpose(-1, -2)

    def basis(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_basis = self.project(features)
        if self.training:
            covariance = self.covariance(raw_basis)
            with torch.no_grad():
                self.feature_cov.mul_(self.momentum).add_(
                    covariance.detach(), alpha=1.0 - self.momentum
                )
        else:
            covariance = self.feature_cov.to(raw_basis)
        return self.whiten(raw_basis, covariance), raw_basis, covariance

    def integral_update(
        self, tokens: torch.Tensor, basis: torch.Tensor
    ) -> torch.Tensor:
        coefficients = torch.matmul(basis.transpose(-1, -2), tokens)
        weighted_basis = basis * F.softplus(self.log_eigenvalues)[None, None]
        return torch.matmul(weighted_basis, coefficients)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(features)
        basis, _raw_basis, _covariance = self.basis(features)
        update = self.integral_update(tokens, basis)
        update = update.transpose(1, 2).reshape_as(features)
        return features + self.residual_gate * update


class OrthogonalAttentionHybridFlowOperator(HybridFlowOperator):
    """Protected-Hybrid challenger with one midpoint orthogonal operator."""

    def __init__(
        self,
        config: ModelConfig,
        psi_dim: int = 16,
        momentum: float = 0.9,
        jitter: float = 1e-4,
    ) -> None:
        super().__init__(config)
        self.orthogonal_attention = OrthogonalIntegralAttention2d(
            config.width,
            psi_dim=psi_dim,
            momentum=momentum,
            jitter=jitter,
        )

    def forward(
        self, flow: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raw = flow
        features, uv_mean, uv_std = self._features(raw)
        hidden = self.lift(features)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.blocks) // 2):
                hidden = self.orthogonal_attention(self.multiscale(hidden))
        hidden = self.final_norm(hidden)

        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (raw[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (
            raw[:, -1:, ..., :2] - raw[:, -5:-4, ..., :2]
        ) / (4.0 * uv_std)
        trend = recent * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base + trend + correction) * uv_std + uv_mean
        prediction = torch.cat(
            (prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1
        )
        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = (
            F.softplus(raw_width)
            * uv_std
            * self.interval_temperature.clamp_min(1e-3)
        )
        half = torch.cat(
            (half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1
        )
        return prediction, half

    def predict(
        self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del metadata
        return self.forward(input_array)
