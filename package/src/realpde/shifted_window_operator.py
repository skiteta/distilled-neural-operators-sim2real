from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ShiftedWindowOperatorConfig:
    """Small scratch-trained scOT-style operator for the 32x64 Track-1 grid."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 64
    depths: tuple[int, ...] = (2, 2, 2)
    heads: tuple[int, ...] = (4, 8, 8)
    window: tuple[int, int] = (4, 4)
    expansion: int = 3
    dropout: float = 0.02
    use_pressure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window_partition(field: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    batch, height, width, channels = field.shape
    wh, ww = window
    if height % wh or width % ww:
        raise ValueError(f"field {(height, width)} is not divisible by window {window}")
    return (
        field.reshape(batch, height // wh, wh, width // ww, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, wh * ww, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window: tuple[int, int],
    height: int,
    width: int,
    batch: int,
) -> torch.Tensor:
    wh, ww = window
    channels = windows.shape[-1]
    return (
        windows.reshape(batch, height // wh, width // ww, wh, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


class CosineWindowAttention(nn.Module):
    """SwinV2-style cosine attention inside fixed non-overlapping windows."""

    def __init__(
        self,
        width: int,
        heads: int,
        window: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.window = window
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.log(torch.full((heads, 1, 1), 10.0)))

        wh, ww = window
        self.relative_bias = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), heads))
        yy, xx = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        coordinates = torch.stack((yy, xx)).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += wh - 1
        relative[..., 1] += ww - 1
        relative[..., 0] *= 2 * ww - 1
        self.register_buffer("relative_index", relative.sum(-1), persistent=True)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        count, length, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(count, length, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        scale = self.logit_scale.clamp(max=torch.log(torch.tensor(100.0, device=tokens.device))).exp()
        logits = (query @ key.transpose(-2, -1)) * scale
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(length, length, self.heads)
        logits = logits + bias.permute(2, 0, 1).unsqueeze(0).to(logits.dtype)
        if mask is not None:
            windows = mask.shape[0]
            batch = count // windows
            logits = logits.reshape(batch, windows, self.heads, length, length)
            logits = logits + mask[None, :, None].to(logits.dtype)
            logits = logits.reshape(count, self.heads, length, length)
        weights = self.dropout(logits.softmax(dim=-1))
        output = (weights @ value).transpose(1, 2).reshape(count, length, self.width)
        return self.output(output)


class ShiftedWindowBlock(nn.Module):
    """Pre-norm shifted-window attention and channel MLP."""

    def __init__(
        self,
        width: int,
        heads: int,
        window: tuple[int, int],
        shift: tuple[int, int],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(width)
        self.attention = CosineWindowAttention(width, heads, window, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * expansion, width),
            nn.Dropout(dropout),
        )
        self.attention_scale = nn.Parameter(torch.full((width,), 1e-2))
        self.mlp_scale = nn.Parameter(torch.full((width,), 1e-2))

    def _mask(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        sh, sw = self.shift
        if sh == 0 and sw == 0:
            return None
        wh, ww = self.window
        labels = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
        value = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                labels[:, h_slice, w_slice, :] = value
                value += 1
        windows = _window_partition(labels, self.window).squeeze(-1)
        differences = windows[:, None, :] - windows[:, :, None]
        return differences.ne(0).to(torch.float32) * -100.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden.shape
        field = hidden.permute(0, 2, 3, 1)
        normalized = self.norm1(field)
        sh, sw = self.shift
        if sh or sw:
            normalized = torch.roll(normalized, shifts=(-sh, -sw), dims=(1, 2))
        windows = _window_partition(normalized, self.window)
        windows = self.attention(windows, self._mask(height, width, hidden.device))
        update = _window_reverse(windows, self.window, height, width, batch)
        if sh or sw:
            update = torch.roll(update, shifts=(sh, sw), dims=(1, 2))
        field = field + self.attention_scale * update
        field = field + self.mlp_scale * self.mlp(self.norm2(field))
        return field.permute(0, 3, 1, 2).contiguous()


class ShiftedWindowStage(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        heads: int,
        window: tuple[int, int],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        shift = (window[0] // 2, window[1] // 2)
        self.blocks = nn.ModuleList(
            ShiftedWindowBlock(
                width, heads, window, (0, 0) if index % 2 == 0 else shift,
                expansion, dropout,
            )
            for index in range(depth)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class ConvNeXtSkip(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.norm = nn.GroupNorm(1, width)
        self.pointwise = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.scale * self.pointwise(self.norm(self.depthwise(hidden)))


class ShiftedWindowFlowOperator(nn.Module):
    """scOT-inspired scratch operator; not a reproduction of POSEIDON."""

    def __init__(self, config: ShiftedWindowOperatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ShiftedWindowOperatorConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("the Track-1 contract is fixed to 20-to-20")
        if len(self.config.depths) != 3 or len(self.config.heads) != 3:
            raise ValueError("scOT-lite is fixed to three resolution levels")
        modalities = 3 if self.config.use_pressure else 2
        descriptor_channels = 4 * modalities
        in_channels = self.config.in_steps * modalities + 2 + descriptor_channels
        widths = (self.config.width, self.config.width * 2, self.config.width * 4)
        self.lift = nn.Conv2d(in_channels, widths[0], 1)
        self.patch_embed = nn.Conv2d(widths[0], widths[0], 2, stride=2)
        self.encoder0 = ShiftedWindowStage(widths[0], self.config.depths[0], self.config.heads[0], self.config.window, self.config.expansion, self.config.dropout)
        self.merge0 = nn.Conv2d(widths[0], widths[1], 2, stride=2)
        self.encoder1 = ShiftedWindowStage(widths[1], self.config.depths[1], self.config.heads[1], self.config.window, self.config.expansion, self.config.dropout)
        self.merge1 = nn.Conv2d(widths[1], widths[2], 2, stride=2)
        self.bottleneck = ShiftedWindowStage(widths[2], self.config.depths[2], self.config.heads[2], self.config.window, self.config.expansion, self.config.dropout)
        self.skip0 = ConvNeXtSkip(widths[0], self.config.dropout)
        self.skip1 = ConvNeXtSkip(widths[1], self.config.dropout)
        self.expand1 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.fuse1 = nn.Conv2d(widths[1] * 2, widths[1], 1)
        self.decoder1 = ShiftedWindowStage(widths[1], self.config.depths[1], self.config.heads[1], self.config.window, self.config.expansion, self.config.dropout)
        self.expand0 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.fuse0 = nn.Conv2d(widths[0] * 2, widths[0], 1)
        self.decoder0 = ShiftedWindowStage(widths[0], self.config.depths[0], self.config.heads[0], self.config.window, self.config.expansion, self.config.dropout)
        self.recover = nn.ConvTranspose2d(widths[0], widths[0], 2, stride=2)
        self.final_norm = nn.GroupNorm(8, widths[0])
        self.velocity_head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], 3, padding=1), nn.GELU(),
            nn.Conv2d(widths[0], self.config.out_steps * 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0] // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(widths[0] // 2, self.config.out_steps * 2, 1),
        )
        self.trend_scale = nn.Parameter(torch.zeros(self.config.out_steps, 2))
        self.interval_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        nn.init.normal_(self.velocity_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.normal_(self.width_head[-1].weight, std=1e-3)
        nn.init.constant_(self.width_head[-1].bias, -2.5)

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def _features(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, 32, 64, 3)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(flow.shape)}")
        uv = flow[..., :2]
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_uv = (uv - uv_mean) / uv_std
        selected = uv
        if self.config.use_pressure:
            pressure = flow[..., 2:3]
            pressure_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
            pressure_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            normalized = torch.cat((normalized_uv, (pressure - pressure_mean) / pressure_std), dim=-1)
            selected = flow
        else:
            normalized = normalized_uv
        batch, _, height, width, _ = flow.shape
        flat = normalized.permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        descriptors = torch.cat((
            selected.mean(dim=(1, 2, 3)), selected.std(dim=(1, 2, 3)),
            selected[:, -1].mean(dim=(1, 2)), selected[:, -1].std(dim=(1, 2)),
        ), dim=-1)[:, :, None, None].expand(batch, -1, height, width)
        coords = self._coordinates(batch, height, width, flow.device, flow.dtype)
        return torch.cat((flat, coords, descriptors), dim=1), uv_mean, uv_std

    def hidden_features(self, flow: torch.Tensor) -> torch.Tensor:
        features, _, _ = self._features(flow)
        hidden = self.patch_embed(self.lift(features))
        encoded0 = self.encoder0(hidden)
        encoded1 = self.encoder1(self.merge0(encoded0))
        hidden = self.bottleneck(self.merge1(encoded1))
        hidden = self.decoder1(self.fuse1(torch.cat((self.expand1(hidden), self.skip1(encoded1)), dim=1)))
        hidden = self.decoder0(self.fuse0(torch.cat((self.expand0(hidden), self.skip0(encoded0)), dim=1)))
        return self.final_norm(self.recover(hidden))

    def forward(self, flow: torch.Tensor, *, return_interval: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_features(flow)
        _, uv_mean, uv_std = self._features(flow)
        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(batch, self.config.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        base = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4.0 * uv_std)
        prediction_uv = (base + recent * self.trend_scale[None, :, None, None, :] + correction) * uv_std + uv_mean
        prediction = torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)
        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(batch, self.config.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        half_uv = F.softplus(raw_width) * uv_std * self.interval_temperature.clamp_min(1e-3)
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        return prediction, half

    def predict(self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del metadata
        return torch.cat([self.forward(input_array[index:index + 1]) for index in range(len(input_array))])
