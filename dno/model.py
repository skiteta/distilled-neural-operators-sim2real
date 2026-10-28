"""The point forecaster: a hybrid spectral/convolutional neural operator with a factorized decoder.

Input:  observed window, float32 (N, 20, H, W, 3) = 20 frames of (u, v, p) in physical units (p is 0 for real data).
Output: forecast (N, 20, H, W, 3) = the next 20 frames (p is 0), and the final hidden field (N, width, H, W) that the
        interval heads read.

Pipeline of one forward pass
  1. per-window standardization and input features (flattened frames, coordinates, window descriptors)
  2. 1x1 lift to `width` channels
  3. `depth` OperatorBlocks (local depthwise conv + low-rank Fourier mix + channel mix, then a gated FFN);
     a two-scale U-Net context is applied after the first half of the blocks
  4. decoding: last observed frame + learned trend + learned 20-frame correction, produced twice (a direction field and
     an amplitude field) and combined so that the predicted fluctuations take the direction of one and the energy of the
     other; a final residual rotates the fluctuation direction at fixed energy.

Parameter names match the training code, so checkpoints written by `package/` load directly (see `load_point_model`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

NORM_FLOOR = 1e-8


@dataclass(frozen=True)
class OperatorConfig:
    in_steps: int = 20
    out_steps: int = 20
    width: int = 128
    depth: int = 4
    spectral_rank: int = 32
    modes_y: int = 12
    modes_x: int = 20
    expansion: int = 3
    dropout: float = 0.03

    @classmethod
    def from_checkpoint(cls, config: dict) -> OperatorConfig:
        if not config.get("use_pressure", True) or config.get("use_physics_features", False):
            raise ValueError(
                "only the deployed configuration (pressure channel on, physics features off) is implemented"
            )
        return cls(**{k: config[k] for k in cls.__dataclass_fields__ if k in config})


class LowRankSpectralMix(nn.Module):
    """Global mixing in Fourier space through a `rank`-channel bottleneck (a low-rank FNO layer).

    The lowest `modes_y` x `modes_x` rfft2 modes are multiplied by learned complex weights per bottleneck channel
    (separate weights for positive and negative vertical frequencies); all other modes are dropped.
    """

    def __init__(self, width: int, rank: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.modes_y, self.modes_x = modes_y, modes_x
        self.to_rank = nn.Conv2d(width, rank, 1, bias=False)
        self.from_rank = nn.Conv2d(rank, width, 1, bias=False)
        self.weight_pos = nn.Parameter(torch.randn(rank, modes_y, modes_x, dtype=torch.cfloat) / rank)
        self.weight_neg = nn.Parameter(torch.randn(rank, modes_y, modes_x, dtype=torch.cfloat) / rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_rank(x)
        height, width = z.shape[-2:]
        spectrum = torch.fft.rfft2(z.float(), norm="ortho")
        my, mx = min(self.modes_y, height // 2), min(self.modes_x, spectrum.shape[-1])
        filtered = torch.zeros_like(spectrum)
        filtered[..., :my, :mx] = spectrum[..., :my, :mx] * self.weight_pos[:, :my, :mx]
        filtered[..., -my:, :mx] = spectrum[..., -my:, :mx] * self.weight_neg[:, :my, :mx]
        return self.from_rank(torch.fft.irfft2(filtered, s=(height, width), norm="ortho").to(x.dtype))


class OperatorBlock(nn.Module):
    """x + s * (local + global + channel mix)(GN(x)), then x + s * FFN(GN(x)); s is a learned per-channel scale."""

    def __init__(self, width: int, rank: int, modes_y: int, modes_x: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.local = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.global_mix = LowRankSpectralMix(width, rank, modes_y, modes_x)
        self.channel_mix = nn.Conv2d(width, width, 1)
        self.norm2 = nn.GroupNorm(8, width)
        hidden = width * expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1), nn.GLU(dim=1), nn.GELU(), nn.Dropout(dropout), nn.Conv2d(hidden, width, 1)
        )
        self.residual_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.residual_scale * (self.local(h) + self.global_mix(h) + self.channel_mix(h))
        return x + self.residual_scale * self.ffn(self.norm2(x))


class MultiScaleContext(nn.Module):
    """Two-scale U-Net context: two strided downsamplings with an OperatorBlock each, decoded back with skips."""

    def __init__(self, c: OperatorConfig) -> None:
        super().__init__()
        w = c.width

        def conv_gn_gelu(cin: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, w, 3, stride=stride, padding=1), nn.GroupNorm(8, w), nn.GELU())

        self.down1 = conv_gn_gelu(w, stride=2)
        self.block1 = OperatorBlock(w, c.spectral_rank, c.modes_y, c.modes_x, c.expansion, c.dropout)
        self.down2 = conv_gn_gelu(w, stride=2)
        self.block2 = OperatorBlock(
            w, c.spectral_rank, max(2, c.modes_y // 2), max(2, c.modes_x // 2), c.expansion, c.dropout
        )
        self.decode1 = conv_gn_gelu(2 * w)
        self.decode0 = conv_gn_gelu(2 * w)
        self.scale = nn.Parameter(torch.tensor(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.block1(self.down1(x))
        e2 = self.block2(self.down2(e1))
        d1 = self.decode1(
            torch.cat((F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), 1)
        )
        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.scale * self.decode0(torch.cat((d0, x), 1))


def _velocity_head(width: int, out_steps: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, out_steps * 2, 1))


def coordinates(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(N, 2, H, W) grid of x and y in [-1, 1]."""
    yy = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def combine_direction_and_amplitude(direction: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """Fluctuations (around the temporal mean) of `direction`, rescaled to the fluctuation energy of `amplitude`.

    Norms are taken per pixel over time and the two velocity components. Identity when both fields are equal.
    """
    centered = direction - direction.mean(dim=1, keepdim=True)
    amplitude_centered = amplitude - amplitude.mean(dim=1, keepdim=True)
    direction_norm = torch.linalg.vector_norm(centered, dim=(1, 4), keepdim=True).clamp_min(NORM_FLOOR)
    amplitude_norm = torch.linalg.vector_norm(amplitude_centered, dim=(1, 4), keepdim=True).clamp_min(NORM_FLOOR)
    return direction + (amplitude_norm / direction_norm - 1) * centered


def rotate_at_fixed_energy(base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Add a residual to the fluctuations of `base`, then restore their per-pixel energy (a rotation, not a rescale)."""
    centered = base - base.mean(dim=1, keepdim=True)
    proposed = centered + residual - residual.mean(dim=1, keepdim=True)
    source_norm = torch.linalg.vector_norm(centered, dim=(1, 4), keepdim=True)
    new_norm = torch.linalg.vector_norm(proposed, dim=(1, 4), keepdim=True)
    rotated = torch.where(new_norm > NORM_FLOOR, proposed * (source_norm / new_norm.clamp_min(NORM_FLOOR)), centered)
    return base + (rotated - centered)


class DistilledFlowOperator(nn.Module):
    """The deployed point model (4 blocks after distillation from a 10-block teacher)."""

    def __init__(self, config: OperatorConfig) -> None:
        super().__init__()
        c = self.config = config
        modalities = 3  # u, v, p
        self.lift = nn.Conv2d(c.in_steps * modalities + 2 + 4 * modalities, c.width, 1)
        self.blocks = nn.ModuleList(
            [
                OperatorBlock(c.width, c.spectral_rank, c.modes_y, c.modes_x, c.expansion, c.dropout)
                for _ in range(c.depth)
            ]
        )
        self.multiscale = MultiScaleContext(c)
        self.final_norm = nn.GroupNorm(8, c.width)
        self.velocity_head = _velocity_head(c.width, c.out_steps)  # direction field
        self.trend_scale = nn.Parameter(torch.zeros(c.out_steps, 2))
        self.amplitude_head = _velocity_head(c.width, c.out_steps)  # amplitude field
        self.amplitude_trend_scale = nn.Parameter(torch.zeros(c.out_steps, 2))
        self.direction_residual_head = _velocity_head(c.width, c.out_steps)  # fixed-energy rotation

    def features(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input features (N, 20*3 + 2 + 12, H, W) and the per-window velocity mean/std used for (de)normalization."""
        if flow.ndim != 5 or flow.shape[1] != self.config.in_steps or flow.shape[-1] != 3:
            raise ValueError(f"expected [N,{self.config.in_steps},H,W,3], got {tuple(flow.shape)}")
        uv, pressure = flow[..., :2], flow[..., 2:3]
        uv_mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        uv_std = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        p_mean = pressure.mean(dim=(1, 2, 3), keepdim=True)
        p_std = pressure.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized = torch.cat(((uv - uv_mean) / uv_std, (pressure - p_mean) / p_std), dim=-1)
        batch, _, height, width, _ = flow.shape
        frames = normalized.permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        descriptors = torch.cat(
            (
                flow.mean(dim=(1, 2, 3)),
                flow.std(dim=(1, 2, 3)),
                flow[:, -1].mean(dim=(1, 2)),
                flow[:, -1].std(dim=(1, 2)),
            ),
            dim=-1,
        )
        descriptors = descriptors[:, :, None, None].expand(-1, -1, height, width)
        grid = coordinates(batch, height, width, flow.device, flow.dtype)
        return torch.cat((frames, grid, descriptors), dim=1), uv_mean, uv_std

    def decode(self, hidden, flow, uv_mean, uv_std, head, trend_scale) -> torch.Tensor:
        """Last observed frame + learned multiple of the recent trend + learned correction, in physical units."""
        batch, _, height, width = hidden.shape
        correction = head(hidden).reshape(batch, self.config.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        last = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent_trend = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4 * uv_std)
        return (last + recent_trend * trend_scale[None, :, None, None, :] + correction) * uv_std + uv_mean

    def forward(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self.features(flow)
        hidden = self.lift(features)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index + 1 == max(1, len(self.blocks) // 2):
                hidden = self.multiscale(hidden)
        hidden = self.final_norm(hidden)
        direction = self.decode(hidden, flow, uv_mean, uv_std, self.velocity_head, self.trend_scale)
        # The amplitude and rotation heads do not train the shared trunk (gradient separation).
        amplitude = self.decode(hidden.detach(), flow, uv_mean, uv_std, self.amplitude_head, self.amplitude_trend_scale)
        batch, _, height, width = hidden.shape
        residual = (
            self.direction_residual_head(hidden.detach())
            .reshape(batch, self.config.out_steps, 2, height, width)
            .permute(0, 1, 3, 4, 2)
            * uv_std
        )
        uv = rotate_at_fixed_energy(combine_direction_and_amplitude(direction, amplitude), residual)
        return torch.cat((uv, torch.zeros_like(uv[..., :1])), dim=-1), hidden


# Training-only checkpoint parameters (a softplus half-width head of the first stage; unused at inference).
UNUSED_CHECKPOINT_PREFIXES = ("width_head.", "interval_temperature")


def load_point_model(path, device="cpu") -> DistilledFlowOperator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DistilledFlowOperator(OperatorConfig.from_checkpoint(checkpoint["model_config"]))
    state = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith(UNUSED_CHECKPOINT_PREFIXES)}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
