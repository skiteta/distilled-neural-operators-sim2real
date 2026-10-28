"""Scale-normalized recurrent block-state energy operator for C488."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockStateEnergyConfig:
    state_channels: int = 5
    input_channels: int = 21
    width: int = 48
    residual_blocks: int = 4
    height: int = 32
    spatial_width: int = 64
    block_frames: int = 5
    observed_blocks: int = 4
    future_blocks: int = 4

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class CircularStateBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


class BlockStateTransitionOperator(nn.Module):
    """One shared two-state transition, recurrently applied to future blocks."""

    def __init__(self, config: BlockStateEnergyConfig = BlockStateEnergyConfig()):
        super().__init__()
        self.config = config
        y = torch.linspace(-1.0, 1.0, config.height)
        x = torch.linspace(-1.0, 1.0, config.spatial_width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("coordinates", torch.stack((yy, xx)), persistent=True)
        self.register_buffer(
            "delta_bounds",
            torch.tensor((0.75, 0.75, 1.50, 1.00, 1.00)).view(1, 5, 1, 1),
            persistent=True,
        )
        self.lift = nn.Conv2d(
            config.input_channels,
            config.width,
            5,
            padding=2,
            padding_mode="circular",
        )
        self.blocks = nn.ModuleList(CircularStateBlock(config.width) for _ in range(config.residual_blocks))
        self.norm = nn.GroupNorm(8, config.width)
        self.output = nn.Conv2d(config.width, config.state_channels, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def mean_derivatives(state: Tensor) -> Tensor:
        means = state[:, :2]
        dy = 0.5 * (torch.roll(means, -1, dims=2) - torch.roll(means, 1, dims=2))
        dx = 0.5 * (torch.roll(means, -1, dims=3) - torch.roll(means, 1, dims=3))
        return torch.cat((dy, dx), dim=1)

    def features(self, previous: Tensor, current: Tensor) -> Tensor:
        if previous.shape != current.shape or previous.shape[1:] != (
            self.config.state_channels,
            self.config.height,
            self.config.spatial_width,
        ):
            raise ValueError(f"unexpected C488 states {list(previous.shape)} {list(current.shape)}")
        coordinates = self.coordinates[None].expand(len(current), -1, -1, -1)
        return torch.cat(
            (previous, current, current - previous, self.mean_derivatives(current), coordinates),
            dim=1,
        )

    def forward(self, previous: Tensor, current: Tensor) -> Tensor:
        hidden = self.lift(self.features(previous, current))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.output(F.gelu(self.norm(hidden)))
        return current + self.delta_bounds * torch.tanh(raw)

    def rollout(self, previous: Tensor, current: Tensor, steps: int = 4) -> Tensor:
        values = []
        for _ in range(steps):
            following = self(previous, current)
            values.append(following)
            previous, current = current, following
        return torch.stack(values, dim=1)


def robust_velocity_scale(observed_velocity: Tensor) -> Tensor:
    """Return q90 speed over observed time/space, shape (B,1,1,1)."""
    speed = observed_velocity[..., :2].float().square().sum(dim=-1).sqrt()
    scale = torch.quantile(speed.flatten(1), 0.90, dim=1).clamp_min(1e-4)
    return scale[:, None, None, None]


def block_states(sequence_velocity: Tensor, scale: Tensor) -> Tensor:
    """Convert Bx(5K)xHxWx2 velocity to BxKx5xHxW physical states."""
    batch, frames, height, width, channels = sequence_velocity.shape
    if frames % 5 or channels < 2:
        raise ValueError(f"unexpected C488 sequence shape {list(sequence_velocity.shape)}")
    blocks = sequence_velocity[..., :2].float().reshape(batch, frames // 5, 5, height, width, 2)
    mean = blocks.mean(dim=2)
    centered = blocks - mean[:, :, None]
    energy = 0.5 * centered.square().sum(dim=-1).mean(dim=2)
    time_axis = sequence_velocity.new_tensor((-2.0, -1.0, 0.0, 1.0, 2.0)).float()
    slope = (blocks * time_axis[None, None, :, None, None, None]).sum(dim=2) / 10.0
    normalized_mean = mean / scale[:, None]
    normalized_slope = slope / scale[:, None]
    log_energy = torch.log(energy / scale[:, None, 0].square() + 1e-6).clamp(-14.0, 4.0)
    return torch.cat(
        (
            normalized_mean.permute(0, 1, 4, 2, 3),
            log_energy[:, :, None],
            normalized_slope.permute(0, 1, 4, 2, 3),
        ),
        dim=2,
    )


def energy_from_future_states(states: Tensor, scale: Tensor) -> Tensor:
    """Exact equal-block law-of-total-variance reconstruction."""
    if states.ndim != 5 or states.shape[2] != 5:
        raise ValueError(f"unexpected C488 future-state shape {list(states.shape)}")
    means = states[:, :, :2].permute(0, 1, 3, 4, 2) * scale[:, None]
    within = (torch.exp(states[:, :, 2].clamp(-14.0, 4.0)) - 1e-6).clamp_min(0.0)
    within = within * scale[:, None, 0].square()
    total_mean = means.mean(dim=1, keepdim=True)
    between = 0.5 * (means - total_mean).square().sum(dim=-1).mean(dim=1)
    return within.mean(dim=1) + between


def normalize_states(states: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (states - mean[:, None]) / std[:, None]


def denormalize_states(states: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return states * std[:, None] + mean[:, None]
