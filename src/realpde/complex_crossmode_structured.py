"""Full-history complex cross-mode backbone for the fixed C487 screen."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ComplexCrossModeConfig:
    channels: int = 8
    frames: int = 20
    height: int = 32
    width: int = 64
    outputs: int = 5
    cutoff: int = 12
    token_width: int = 64
    heads: int = 4
    layers: int = 3
    feedforward_width: int = 192
    local_width: int = 64
    local_blocks: int = 3

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ModeInteractionBlock(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_width: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Linear(feedforward_width, width),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.norm_feedforward(tokens))


class CircularResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = min(8, width)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, field: Tensor) -> Tensor:
        return field + self.net(field)


class ComplexCrossModeBackbone(nn.Module):
    """Maps 20x8 observed fields to 20x5 raw structural coordinates."""

    def __init__(self, config: ComplexCrossModeConfig = ComplexCrossModeConfig()):
        super().__init__()
        self.config = config
        ky = torch.fft.fftfreq(config.height) * config.height
        kx = torch.fft.rfftfreq(config.width) * config.width
        radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
        mask = radius <= float(config.cutoff)
        mode_y, mode_x = mask.nonzero(as_tuple=True)
        mode_ky = ky[mode_y]
        mode_kx = kx[mode_x]
        mode_radius = radius[mode_y, mode_x]
        angle = torch.atan2(mode_ky, mode_kx)
        angle = torch.where(mode_radius > 0, angle, torch.zeros_like(angle))
        coordinates = torch.stack(
            (
                mode_ky / config.cutoff,
                mode_kx / config.cutoff,
                mode_radius / config.cutoff,
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=-1,
        )
        self.register_buffer("mode_y", mode_y.long(), persistent=True)
        self.register_buffer("mode_x", mode_x.long(), persistent=True)
        self.register_buffer("mode_coordinates", coordinates.float(), persistent=True)

        trajectory_width = config.frames * config.channels * 2
        self.token_projection = nn.Sequential(
            nn.LayerNorm(trajectory_width),
            nn.Linear(trajectory_width, config.token_width),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(5, config.token_width),
            nn.GELU(),
            nn.Linear(config.token_width, config.token_width),
        )
        self.mode_blocks = nn.ModuleList(
            ModeInteractionBlock(config.token_width, config.heads, config.feedforward_width)
            for _ in range(config.layers)
        )
        self.final_mode_norm = nn.LayerNorm(config.token_width)
        self.horizon_embedding = nn.Parameter(torch.empty(config.frames, config.token_width))
        self.spectral_head = nn.Sequential(
            nn.Linear(config.token_width, config.feedforward_width),
            nn.GELU(),
            nn.Linear(config.feedforward_width, config.outputs * 2),
        )

        local_inputs = config.frames * config.channels
        self.local_lift = nn.Conv2d(
            local_inputs,
            config.local_width,
            5,
            padding=2,
            padding_mode="circular",
        )
        self.local_blocks = nn.ModuleList(
            CircularResidualBlock(config.local_width) for _ in range(config.local_blocks)
        )
        self.local_film = nn.Linear(config.token_width, config.local_width * 2)
        self.local_norm = nn.GroupNorm(min(8, config.local_width), config.local_width)
        self.local_head = nn.Conv2d(
            config.local_width,
            config.frames * config.outputs,
            1,
        )
        self.reset_parameters()

    @property
    def mode_count(self) -> int:
        return int(self.mode_y.numel())

    def reset_parameters(self) -> None:
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        spectral_output = self.spectral_head[-1]
        assert isinstance(spectral_output, nn.Linear)
        nn.init.normal_(spectral_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(spectral_output.bias)
        nn.init.normal_(self.local_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.local_head.bias)

    def complex_tokens(self, features: Tensor) -> tuple[Tensor, Tensor]:
        config = self.config
        if list(features.shape[1:]) != [config.frames, config.height, config.width, config.channels]:
            raise ValueError(f"unexpected C487 feature shape {list(features.shape)}")
        spectrum = torch.fft.rfft2(
            features.permute(0, 1, 4, 2, 3).float(),
            dim=(-2, -1),
            norm="ortho",
        )
        selected = spectrum[..., self.mode_y, self.mode_x]
        selected = selected.permute(0, 3, 1, 2)
        trajectory = torch.view_as_real(selected).reshape(features.shape[0], self.mode_count, -1)
        tokens = self.token_projection(trajectory)
        tokens = tokens + self.coordinate_projection(self.mode_coordinates)[None]
        return tokens, selected

    def spectral_raw(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        config = self.config
        for block in self.mode_blocks:
            tokens = block(tokens)
        tokens = self.final_mode_norm(tokens)
        future_tokens = tokens[:, :, None, :] + self.horizon_embedding[None, None, :, :]
        complex_values = self.spectral_head(future_tokens)
        complex_values = complex_values.reshape(
            tokens.shape[0],
            self.mode_count,
            config.frames,
            config.outputs,
            2,
        ).contiguous()
        complex_values = torch.view_as_complex(complex_values)
        lattice = torch.zeros(
            tokens.shape[0],
            config.frames,
            config.outputs,
            config.height,
            config.width // 2 + 1,
            dtype=complex_values.dtype,
            device=complex_values.device,
        )
        lattice[..., self.mode_y, self.mode_x] = complex_values.permute(0, 2, 3, 1)
        raw = torch.fft.irfft2(
            lattice,
            s=(config.height, config.width),
            dim=(-2, -1),
            norm="ortho",
        ).permute(0, 1, 3, 4, 2)
        return raw, tokens

    def local_raw(self, features: Tensor, tokens: Tensor) -> Tensor:
        config = self.config
        local = features.permute(0, 1, 4, 2, 3).reshape(
            features.shape[0],
            config.frames * config.channels,
            config.height,
            config.width,
        )
        local = self.local_lift(local)
        for block in self.local_blocks:
            local = block(local)
        scale, shift = self.local_film(tokens.mean(dim=1)).chunk(2, dim=-1)
        local = self.local_norm(local)
        local = local * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None])
        local = local + 0.1 * shift[:, :, None, None]
        local = torch.nn.functional.gelu(local)
        raw = self.local_head(local).reshape(
            features.shape[0],
            config.frames,
            config.outputs,
            config.height,
            config.width,
        )
        return raw.permute(0, 1, 3, 4, 2)

    def forward_components(self, features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        tokens, selected = self.complex_tokens(features)
        spectral, mixed_tokens = self.spectral_raw(tokens)
        local = self.local_raw(features, mixed_tokens)
        raw = spectral + local
        return raw, {
            "input_complex_modes": selected,
            "mixed_tokens": mixed_tokens,
            "spectral_raw": spectral,
            "local_raw": local,
        }

    def forward(self, features: Tensor) -> Tensor:
        raw, _ = self.forward_components(features)
        return raw
