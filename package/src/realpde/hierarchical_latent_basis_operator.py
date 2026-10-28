from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HierarchicalLatentBasisConfig:
    """Fixed LSM-lite configuration for the C304 mechanism screen."""

    in_steps: int = 20
    out_steps: int = 20
    width: int = 96
    depth: int = 6
    bases: int = 8
    scales: tuple[int, ...] = (1, 2, 4)
    expansion: int = 2
    dropout: float = 0.02
    use_pressure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveLatentBasis(nn.Module):
    """Project a field to learned global bases, mix coefficients, reconstruct."""

    def __init__(self, width: int, bases: int) -> None:
        super().__init__()
        self.width = width
        self.bases = bases
        self.value = nn.Conv2d(width, width, 1, bias=False)
        self.basis_logits = nn.Conv2d(width, bases, 1)
        self.coefficient_norm = nn.LayerNorm(width)
        self.coefficient_mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.basis_mix = nn.Parameter(torch.eye(bases) + 0.01 * torch.randn(bases, bases))
        self.output = nn.Conv2d(width, width, 1, bias=False)

    def basis_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.basis_logits(hidden)
        batch, bases, height, width = logits.shape
        return logits.flatten(2).softmax(dim=-1).reshape(batch, bases, height, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        values = self.value(hidden)
        weights = self.basis_weights(hidden)
        coefficients = torch.einsum("bchw,bkhw->bkc", values, weights)
        coefficients = coefficients + self.coefficient_mlp(
            self.coefficient_norm(coefficients)
        )
        mixing = self.basis_mix.softmax(dim=-1)
        coefficients = torch.einsum("kj,bjc->bkc", mixing, coefficients)
        reconstructed = torch.einsum("bkc,bkhw->bchw", coefficients, weights)
        return self.output(reconstructed)


class HierarchicalLatentBasisBlock(nn.Module):
    """Parallel multi-resolution latent-basis operators plus channel update."""

    def __init__(
        self,
        width: int,
        bases: int,
        scales: tuple[int, ...],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.scales = scales
        self.norm1 = nn.GroupNorm(groups, width)
        self.latent = nn.ModuleList(
            AdaptiveLatentBasis(width, bases) for _ in scales
        )
        self.fuse = nn.Conv2d(width * len(scales), width, 1, bias=False)
        self.channel = nn.Conv2d(width, width, 1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.operator_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))
        self.ffn_scale = nn.Parameter(torch.full((1, width, 1, 1), 1e-2))

    def scale_updates(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        height, width = hidden.shape[-2:]
        updates = []
        for factor, operator in zip(self.scales, self.latent, strict=True):
            pooled = hidden if factor == 1 else F.avg_pool2d(hidden, factor)
            update = operator(pooled)
            if factor != 1:
                update = F.interpolate(
                    update, size=(height, width), mode="bilinear", align_corners=False
                )
            updates.append(update)
        return updates

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        update = self.fuse(torch.cat(self.scale_updates(normalized), dim=1))
        update = update + self.channel(normalized)
        hidden = hidden + self.operator_scale * update
        return hidden + self.ffn_scale * self.ffn(self.norm2(hidden))


class HierarchicalLatentBasisFlowOperator(nn.Module):
    """LSM-inspired minimal hierarchical multiple-basis flow operator."""

    def __init__(self, config: HierarchicalLatentBasisConfig | None = None) -> None:
        super().__init__()
        self.config = config or HierarchicalLatentBasisConfig()
        if self.config.in_steps != 20 or self.config.out_steps != 20:
            raise ValueError("C304 is fixed to the official 20-to-20 contract")
        if not self.config.scales or any(32 % scale or 64 % scale for scale in self.config.scales):
            raise ValueError("all scales must divide the official 32x64 grid")
        modalities = 3 if self.config.use_pressure else 2
        descriptor_channels = 4 * modalities
        in_channels = self.config.in_steps * modalities + 2 + descriptor_channels
        self.lift = nn.Conv2d(in_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            HierarchicalLatentBasisBlock(
                self.config.width,
                self.config.bases,
                self.config.scales,
                self.config.expansion,
                self.config.dropout,
            )
            for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, self.config.width)
        self.velocity_head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width, self.config.out_steps * 2, 1),
        )
        self.width_head = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.width // 2, self.config.out_steps * 2, 1),
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
        descriptors = torch.cat(
            (
                selected.mean(dim=(1, 2, 3)),
                selected.std(dim=(1, 2, 3)),
                selected[:, -1].mean(dim=(1, 2)),
                selected[:, -1].std(dim=(1, 2)),
            ),
            dim=-1,
        )[:, :, None, None].expand(batch, -1, height, width)
        coords = self._coordinates(batch, height, width, flow.device, flow.dtype)
        return torch.cat((flat, coords, descriptors), dim=1), uv_mean, uv_std

    def forward(self, flow: torch.Tensor, *, return_interval: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features, uv_mean, uv_std = self._features(flow)
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        batch, _, height, width = hidden.shape
        correction = self.velocity_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        base = (flow[:, -1:, ..., :2] - uv_mean) / uv_std
        recent = (flow[:, -1:, ..., :2] - flow[:, -5:-4, ..., :2]) / (4.0 * uv_std)
        trend = recent * self.trend_scale[None, :, None, None, :]
        prediction_uv = (base + trend + correction) * uv_std + uv_mean
        prediction = torch.cat((prediction_uv, torch.zeros_like(prediction_uv[..., :1])), dim=-1)
        if not return_interval:
            return prediction
        raw_width = self.width_head(hidden).reshape(
            batch, self.config.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        half_uv = F.softplus(raw_width) * uv_std * self.interval_temperature.clamp_min(1e-3)
        half = torch.cat((half_uv, torch.zeros_like(half_uv[..., :1])), dim=-1)
        return prediction, half

    def predict(self, input_array: torch.Tensor, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        del metadata
        outputs = [self.forward(input_array[index:index + 1]) for index in range(len(input_array))]
        return torch.cat(outputs)
