from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DirectSpaceTimeConfig:
    in_steps: int = 20
    out_steps: int = 20
    width: int = 48
    depth: int = 6
    expansion: int = 2
    dropout: float = 0.02
    maximum_normalized_correction: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


class SeparableSpaceTimeBlock(nn.Module):
    """Shared local transport and temporal-frequency mixing on an explicit time axis."""

    def __init__(self, width: int, expansion: int, dropout: float, dilation: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.temporal = nn.Conv3d(
            width, width, (3, 1, 1), padding=(1, 0, 0), groups=width
        )
        self.spatial = nn.Conv3d(
            width,
            width,
            (1, 3, 3),
            padding=(0, dilation, dilation),
            dilation=(1, dilation, dilation),
            groups=width,
        )
        self.channel = nn.Conv3d(width, width, 1)
        self.frequency_weight = nn.Parameter(
            torch.zeros(width, 11, 2, dtype=torch.float32)
        )
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def _temporal_spectral(self, value: torch.Tensor) -> torch.Tensor:
        source_dtype = value.dtype
        spectrum = torch.fft.rfft(value.float(), dim=2, norm="ortho")
        bins = spectrum.shape[2]
        weight = torch.view_as_complex(
            self.frequency_weight[:, :bins].contiguous()
        )[None, :, :, None, None]
        return torch.fft.irfft(
            spectrum * weight, n=value.shape[2], dim=2, norm="ortho"
        ).to(source_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        update = (
            self.temporal(normalized)
            + self.spatial(normalized)
            + self.channel(normalized)
            + self._temporal_spectral(normalized)
        )
        value = value + self.scale * update
        return value + self.scale * self.ffn(self.norm2(value))


class DirectSeparableSpaceTimeOperator(nn.Module):
    """Deterministic direct 20-to-20 forecaster that never flattens time into channels."""

    def __init__(self, config: DirectSpaceTimeConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectSpaceTimeConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C102 is fixed to the official 20-to-20 contract")
        width = self.config.width
        # Normalized u/v/p plus physical time and x/y coordinates.
        self.lift = nn.Conv3d(6, width, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        dilations = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            SeparableSpaceTimeBlock(
                width, self.config.expansion, self.config.dropout, dilations[index]
            )
            for index in range(self.config.depth)
        )
        self.time_projection = nn.Parameter(torch.empty(self.config.out_steps, self.config.in_steps))
        nn.init.orthogonal_(self.time_projection)
        self.future_embedding = nn.Parameter(
            torch.empty(1, width, self.config.out_steps, 1, 1)
        )
        nn.init.normal_(self.future_embedding, std=0.02)
        groups = 8 if width % 8 == 0 else 1
        self.decode_norm = nn.GroupNorm(groups, width)
        self.head = nn.Sequential(
            nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(width, 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _coordinates(source: torch.Tensor) -> torch.Tensor:
        batch, _, steps, height, width = source.shape
        time = torch.linspace(-1.0, 0.0, steps, device=source.device, dtype=source.dtype)
        time = time[None, None, :, None, None].expand(batch, 1, steps, height, width)
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        spatial = torch.stack((grid_x, grid_y), dim=0)
        spatial = spatial[None, :, None].expand(batch, 2, steps, height, width)
        return torch.cat((time, spatial), dim=1)

    @staticmethod
    def _normalize(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv = flow[..., :2].float()
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_uv = (uv - uv_mean) / uv_std
        pressure = flow[..., 2:3].float()
        pressure_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
        pressure_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_pressure = (pressure - pressure_mean) / pressure_std
        normalized = torch.cat((normalized_uv, normalized_pressure), dim=-1)
        return normalized, uv_mean, uv_std

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        normalized, uv_mean, uv_std = self._normalize(flow)
        source = normalized.permute(0, 4, 1, 2, 3).contiguous()
        source = torch.cat((source, self._coordinates(source)), dim=1)
        hidden = self.lift(source)
        for block in self.blocks:
            hidden = block(hidden)
        # The same learned temporal map is shared at every channel and spatial location.
        hidden = torch.einsum("bcthw,ot->bcohw", hidden, self.time_projection)
        hidden = hidden + self.future_embedding
        hidden = F.interpolate(
            hidden,
            size=(self.config.out_steps, flow.shape[2], flow.shape[3]),
            mode="trilinear",
            align_corners=False,
        )
        raw = self.head(self.decode_norm(hidden)).permute(0, 2, 3, 4, 1)
        correction = self.config.maximum_normalized_correction * torch.tanh(raw)
        last = normalized[:, -1:, ..., :2]
        recent = (normalized[:, -1:, ..., :2] - normalized[:, -5:-4, ..., :2]) / 4.0
        trend = recent * self.trend_scale[None, :, None, None, :]
        prediction_uv = (last + trend + correction) * uv_std + uv_mean
        return torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)
