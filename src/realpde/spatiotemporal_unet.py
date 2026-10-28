"""Dependency-free RealPDEBench spatiotemporal U-Net.

This is a clean local reimplementation of the architecture in
``realpdebench/model/unet.py`` at commit
``62f4c80ab17f78933d046f2b038531dbc6a478a0`` (CC-BY-NC-4.0).  It deliberately
does not import the reference repository or its einops/rotary dependencies, so
the packaged model has a small, auditable dependency surface.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatiotemporalUNetConfig:
    dim: int = 32
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 3
    out_channels: int = 3
    in_time: int = 20
    out_time: int = 20
    attn_heads: int = 4
    attn_dim_head: int = 32
    init_kernel_size: int = 7
    resnet_groups: int = 8
    use_sparse_linear_attn: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dim_mults"] = list(self.dim_mults)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatiotemporalUNetConfig":
        fields = dict(value)
        fields["dim_mults"] = tuple(fields.get("dim_mults", (1, 2, 4)))
        return cls(**fields)


class RelativePositionBias(nn.Module):
    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _bucket(relative_position: Tensor, num_buckets: int, max_distance: int) -> Tensor:
        n = -relative_position
        half = num_buckets // 2
        result = (n < 0).long() * half
        n = n.abs()
        max_exact = half // 2
        is_small = n < max_exact
        safe_n = n.float().clamp_min(float(max_exact))
        large = max_exact + (
            torch.log(safe_n / max_exact)
            / math.log(max_distance / max_exact)
            * (half - max_exact)
        ).long()
        large = large.clamp_max(half - 1)
        return result + torch.where(is_small, n, large)

    def forward(self, length: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, dtype=torch.long, device=device)
        relative = positions[None, :] - positions[:, None]
        buckets = self._bucket(relative, self.num_buckets, self.max_distance)
        return self.relative_attention_bias(buckets).permute(2, 0, 1)


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = ChannelLayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return x + self.fn(x, **kwargs)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, values: Tensor) -> Tensor:
        half = self.dim // 2
        exponent = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=values.device, dtype=values.dtype) * -exponent
        )
        phase = values[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1.0) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
        scale_shift = None
        if self.mlp is not None:
            if time_emb is None:
                raise ValueError("time embedding is required")
            embedded = self.mlp(time_emb)[:, :, None, None, None]
            scale_shift = embedded.chunk(2, dim=1)
        return self.block2(self.block1(x, scale_shift=scale_shift)) + self.res_conv(x)


def _rotate_half(x: Tensor) -> Tensor:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_rotary(x: Tensor, rotary_dim: int) -> Tensor:
    rotary_dim = min(int(rotary_dim), x.shape[-1])
    rotary_dim -= rotary_dim % 2
    if rotary_dim == 0:
        return x
    length = x.shape[-2]
    dtype = x.dtype
    position = torch.arange(length, device=x.device, dtype=torch.float32)
    inverse = 1.0 / (
        10000.0
        ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    phase = torch.einsum("n,d->nd", position, inverse).repeat_interleave(2, dim=-1)
    shape = (1,) * (x.ndim - 2) + phase.shape
    cosine = phase.cos().reshape(shape).to(dtype)
    sine = phase.sin().reshape(shape).to(dtype)
    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cosine + _rotate_half(rotated) * sine
    return torch.cat((rotated, remainder), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, rotary_dim: int = 0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.rotary_dim = int(rotary_dim)
        hidden = heads * dim_head
        self.to_qkv = nn.Linear(dim, hidden * 3, bias=False)
        self.to_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor, pos_bias: Tensor | None = None) -> Tensor:
        *prefix, length, _channels = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        target_shape = (*prefix, length, self.heads, self.dim_head)
        q = q.reshape(target_shape).transpose(-3, -2) * self.scale
        k = k.reshape(target_shape).transpose(-3, -2)
        v = v.reshape(target_shape).transpose(-3, -2)
        if self.rotary_dim:
            q = _apply_rotary(q, self.rotary_dim)
            k = _apply_rotary(k, self.rotary_dim)
        similarity = torch.einsum("...hid,...hjd->...hij", q, k)
        if pos_bias is not None:
            similarity = similarity + pos_bias
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        weights = similarity.softmax(dim=-1)
        output = torch.einsum("...hij,...hjd->...hid", weights, v)
        output = output.transpose(-3, -2).reshape(*prefix, length, -1)
        return self.to_out(output)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.attention = Attention(
            dim, heads=heads, dim_head=dim_head, rotary_dim=min(32, dim_head)
        )

    def forward(self, x: Tensor, pos_bias: Tensor | None = None) -> Tensor:
        batch, channels, frames, height, width = x.shape
        values = x.permute(0, 3, 4, 2, 1)
        values = self.attention(values, pos_bias=pos_bias)
        return values.permute(0, 4, 3, 1, 2).reshape(batch, channels, frames, height, width)


class FullSpatialAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.attention = Attention(dim, heads=heads, dim_head=dim_head)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, height, width = x.shape
        values = x.permute(0, 2, 3, 4, 1).reshape(batch, frames, height * width, channels)
        values = self.attention(values)
        return values.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, height, width = x.shape
        values = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        q, k, v = self.to_qkv(values).chunk(3, dim=1)

        def heads(value: Tensor) -> Tensor:
            return value.reshape(batch * frames, self.heads, self.dim_head, height * width)

        q, k, v = heads(q), heads(k), heads(v)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        output = torch.einsum("bhde,bhdn->bhen", context, q)
        output = output.reshape(batch * frames, self.heads * self.dim_head, height, width)
        output = self.to_out(output)
        return output.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def downsample(dim: int) -> nn.Module:
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def upsample(dim: int) -> nn.Module:
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class SpatiotemporalUNet(nn.Module):
    """Parallel 20-to-20 video U-Net with attention over space and time."""

    def __init__(self, config: SpatiotemporalUNetConfig | None = None):
        super().__init__()
        self.config = config or SpatiotemporalUNetConfig()
        cfg = self.config
        if cfg.init_kernel_size % 2 != 1:
            raise ValueError("init_kernel_size must be odd")
        if cfg.out_time < cfg.in_time or cfg.out_time % cfg.in_time != 0:
            raise ValueError("out_time must be an integer multiple of in_time")

        temporal: Callable[[int], nn.Module] = lambda channels: TemporalAttention(
            channels, cfg.attn_heads, cfg.attn_dim_head
        )
        self.time_rel_pos_bias = RelativePositionBias(
            heads=cfg.attn_heads, max_distance=32
        )
        padding = cfg.init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            cfg.channels,
            cfg.dim,
            cfg.init_kernel_size,
            padding=padding,
        )
        self.init_temporal_attn = Residual(PreNorm(cfg.dim, temporal(cfg.dim)))

        dims = [cfg.dim, *(cfg.dim * multiplier for multiplier in cfg.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = cfg.dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(cfg.dim),
            nn.Linear(cfg.dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        block = partial(ResnetBlock, groups=cfg.resnet_groups)
        conditioned = partial(block, time_emb_dim=time_dim)

        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            last = index == len(in_out) - 1
            spatial = (
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, cfg.attn_heads, cfg.attn_dim_head)))
                if cfg.use_sparse_linear_attn
                else nn.Identity()
            )
            self.downs.append(
                nn.ModuleList(
                    [
                        conditioned(dim_in, dim_out),
                        conditioned(dim_out, dim_out),
                        spatial,
                        Residual(PreNorm(dim_out, temporal(dim_out))),
                        nn.Identity() if last else downsample(dim_out),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = conditioned(mid_dim, mid_dim)
        self.mid_spatial_attn = Residual(
            PreNorm(mid_dim, FullSpatialAttention(mid_dim, cfg.attn_heads, cfg.attn_dim_head))
        )
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal(mid_dim)))
        self.mid_block2 = conditioned(mid_dim, mid_dim)

        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            last = index == len(in_out) - 1
            spatial = (
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, cfg.attn_heads, cfg.attn_dim_head)))
                if cfg.use_sparse_linear_attn
                else nn.Identity()
            )
            self.ups.append(
                nn.ModuleList(
                    [
                        conditioned(dim_out * 2, dim_in),
                        conditioned(dim_in, dim_in),
                        spatial,
                        Residual(PreNorm(dim_in, temporal(dim_in))),
                        nn.Identity() if last else upsample(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            block(cfg.dim * 2, cfg.dim),
            nn.Conv3d(cfg.dim, cfg.out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[-1] != self.config.channels:
            raise ValueError(f"expected [B,T,H,W,{self.config.channels}], got {tuple(x.shape)}")
        if x.shape[1] != self.config.in_time:
            raise ValueError(f"expected {self.config.in_time} input frames, got {x.shape[1]}")
        x = x.permute(0, 4, 1, 2, 3)
        if self.config.out_time > x.shape[2]:
            x = x.repeat(1, 1, self.config.out_time // x.shape[2], 1, 1)
        x = self.init_conv(x)
        bias = self.time_rel_pos_bias(self.config.out_time, x.device)
        x = self.init_temporal_attn(x, pos_bias=bias)
        initial = x.clone()
        time = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        embedding = self.time_mlp(time)
        skips: list[Tensor] = []
        for block1, block2, spatial, temporal, resize in self.downs:
            x = block1(x, embedding)
            x = block2(x, embedding)
            x = spatial(x)
            x = temporal(x, pos_bias=bias)
            skips.append(x)
            x = resize(x)
        x = self.mid_block1(x, embedding)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=bias)
        x = self.mid_block2(x, embedding)
        for block1, block2, spatial, temporal, resize in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, embedding)
            x = block2(x, embedding)
            x = spatial(x)
            x = temporal(x, pos_bias=bias)
            x = resize(x)
        x = self.final_conv(torch.cat((x, initial), dim=1))
        return x.permute(0, 2, 3, 4, 1)

    def train_loss(self, observed: Tensor, target: Tensor) -> Tensor:
        return (self(observed) - target).square().reshape(observed.shape[0], -1).mean(dim=1)
