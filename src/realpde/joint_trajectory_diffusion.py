from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointTrajectoryDiffusionConfig:
    width: int = 32
    depth: int = 6
    expansion: int = 2
    diffusion_steps: int = 100
    sample_steps: int = 8
    ensemble_size: int = 4
    sample_seed: int = 20260813

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalSpectralBlock(nn.Module):
    """Local space/time transport plus an explicit temporal Fourier update."""

    def __init__(self, width: int, expansion: int, max_frequency: int = 21) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        hidden = width * expansion
        self.norm1 = nn.GroupNorm(groups, width)
        self.temporal = nn.Conv3d(
            width, width, (3, 1, 1), padding=(1, 0, 0), groups=width
        )
        self.spatial = nn.Conv3d(
            width, width, (1, 3, 3), padding=(0, 1, 1), groups=width
        )
        self.channel = nn.Conv3d(width, width, 1)
        self.frequency_weight = nn.Parameter(
            torch.zeros(width, max_frequency, 2, dtype=torch.float32)
        )
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, hidden * 2, 1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Conv3d(hidden, width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1, 1), 1e-2))

    def _spectral(self, value: torch.Tensor) -> torch.Tensor:
        source_dtype = value.dtype
        spectrum = torch.fft.rfft(value.float(), dim=2, norm="ortho")
        bins = spectrum.shape[2]
        weight = torch.view_as_complex(
            self.frequency_weight[:, :bins].contiguous()
        )[None, :, :, None, None]
        update = torch.fft.irfft(
            spectrum * weight, n=value.shape[2], dim=2, norm="ortho"
        )
        return update.to(source_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        update = (
            self.temporal(normalized)
            + self.spatial(normalized)
            + self.channel(normalized)
            + self._spectral(normalized)
        )
        value = value + self.scale * update
        return value + self.scale * self.ffn(self.norm2(value))


class JointTrajectoryDenoiser(nn.Module):
    """Denoise a 40-frame joint path while the observed prefix stays clamped."""

    def __init__(self, config: JointTrajectoryDiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or JointTrajectoryDiffusionConfig()
        self.time_channels = 8
        # u/v, observed-prefix mask, normalized physical time, x/y, noise time.
        input_channels = 2 + 1 + 1 + 2 + self.time_channels
        self.lift = nn.Conv3d(input_channels, self.config.width, 1)
        self.blocks = nn.ModuleList(
            TemporalSpectralBlock(self.config.width, self.config.expansion)
            for _ in range(self.config.depth)
        )
        groups = 8 if self.config.width % 8 == 0 else 1
        self.head = nn.Sequential(
            nn.GroupNorm(groups, self.config.width),
            nn.Conv3d(self.config.width, self.config.width, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(self.config.width, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _noise_embedding(self, noise_time: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        half = self.time_channels // 2
        frequency = 2.0 ** torch.arange(
            half, device=noise_time.device, dtype=noise_time.dtype
        )
        phase = math.pi * noise_time[:, None] * frequency[None]
        embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        return embedding[:, :, None, None, None].expand(-1, -1, *shape)

    @staticmethod
    def _coordinates(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, steps, height, width = source.shape
        time = torch.linspace(-1.0, 1.0, steps, device=source.device, dtype=source.dtype)
        time = time[None, None, :, None, None].expand(batch, 1, steps, height, width)
        yy = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        spatial = torch.stack((grid_y, grid_x), dim=0)
        spatial = spatial[None, :, None].expand(batch, 2, steps, height, width)
        return time, spatial

    def forward(
        self,
        observed_normalized: torch.Tensor,
        noisy_future: torch.Tensor,
        noise_time: torch.Tensor,
    ) -> torch.Tensor:
        if observed_normalized.shape[1:] != (20, 32, 64, 2):
            raise ValueError(f"unexpected observed shape {tuple(observed_normalized.shape)}")
        if noisy_future.shape[1:] != (20, 32, 64, 2):
            raise ValueError(f"unexpected future shape {tuple(noisy_future.shape)}")
        trajectory = torch.cat((observed_normalized, noisy_future), dim=1)
        trajectory = trajectory.permute(0, 4, 1, 2, 3).contiguous()
        batch, _, steps, height, width = trajectory.shape
        mask = trajectory.new_zeros((batch, 1, steps, height, width))
        mask[:, :, :20] = 1.0
        physical_time, spatial = self._coordinates(trajectory)
        noise = self._noise_embedding(noise_time, (steps, height, width))
        hidden = self.lift(
            torch.cat((trajectory, mask, physical_time, spatial, noise), dim=1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        prediction = self.head(hidden)[:, :, 20:]
        return prediction.permute(0, 2, 3, 4, 1).contiguous()


class JointTrajectoryDiffusion(nn.Module):
    def __init__(self, config: JointTrajectoryDiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or JointTrajectoryDiffusionConfig()
        self.denoiser = JointTrajectoryDenoiser(self.config)
        beta = torch.linspace(1e-4, 0.02, self.config.diffusion_steps)
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)
        generator = torch.Generator(device="cpu").manual_seed(self.config.sample_seed)
        half = (self.config.ensemble_size + 1) // 2
        noise = torch.randn(half, 20, 32, 64, 2, generator=generator)
        bank = torch.stack((noise, -noise), dim=1).flatten(0, 1)
        self.register_buffer(
            "noise_bank", bank[: self.config.ensemble_size], persistent=True
        )

    @staticmethod
    def normalize(
        observed: torch.Tensor, future: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        uv = observed[..., :2].float()
        mean = uv.mean(dim=(1, 2, 3), keepdim=True)
        scale = uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        normalized_observed = (uv - mean) / scale
        normalized_future = None
        if future is not None:
            normalized_future = (future[..., :2].float() - mean) / scale
        return normalized_observed, normalized_future, mean, scale

    def training_loss(
        self, observed: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observed_n, future_n, _, _ = self.normalize(observed, future)
        assert future_n is not None
        batch = observed.shape[0]
        index = torch.randint(
            0, self.config.diffusion_steps, (batch,), device=observed.device
        )
        alpha = self.alpha_bar[index].to(future_n.dtype)
        shaped = alpha[:, None, None, None, None]
        noise = torch.randn_like(future_n)
        noisy = shaped.sqrt() * future_n + (1.0 - shaped).sqrt() * noise
        predicted_noise = self.denoiser(
            observed_n, noisy, index.to(future_n.dtype) / (self.config.diffusion_steps - 1)
        )
        denoise = F.mse_loss(predicted_noise, noise)
        reconstructed = (
            noisy - (1.0 - shaped).sqrt() * predicted_noise
        ) / shaped.sqrt().clamp_min(1e-4)
        reconstructed = reconstructed.clamp(-8.0, 8.0)
        predicted_spectrum = torch.fft.rfft(reconstructed.float(), dim=1, norm="ortho")
        target_spectrum = torch.fft.rfft(future_n.float(), dim=1, norm="ortho")
        spectral = F.smooth_l1_loss(
            torch.view_as_real(predicted_spectrum),
            torch.view_as_real(target_spectrum),
            beta=0.25,
        )
        predicted_tke = 0.5 * (
            reconstructed - reconstructed.mean(dim=1, keepdim=True)
        ).square().sum(dim=-1).mean(dim=1)
        target_tke = 0.5 * (
            future_n - future_n.mean(dim=1, keepdim=True)
        ).square().sum(dim=-1).mean(dim=1)
        tke = F.smooth_l1_loss(
            torch.log1p(predicted_tke), torch.log1p(target_tke), beta=0.10
        )
        total = denoise + 0.10 * spectral + 0.05 * tke
        return total, {"denoise": denoise, "spectral": spectral, "tke": tke}

    @torch.inference_mode()
    def sample(self, observed: torch.Tensor) -> torch.Tensor:
        observed_n, _, mean, scale = self.normalize(observed)
        batch = observed.shape[0]
        ensemble = self.config.ensemble_size
        repeated_observed = observed_n[:, None].expand(
            -1, ensemble, -1, -1, -1, -1
        ).reshape(batch * ensemble, 20, 32, 64, 2)
        current = self.noise_bank[None].expand(
            batch, -1, -1, -1, -1, -1
        ).reshape(batch * ensemble, 20, 32, 64, 2).to(observed.device)
        timesteps = torch.linspace(
            self.config.diffusion_steps - 1,
            0,
            self.config.sample_steps,
            device=observed.device,
        ).round().long()
        for position, step in enumerate(timesteps):
            index = current.new_full((len(current),), float(step))
            predicted_noise = self.denoiser(
                repeated_observed,
                current,
                index / (self.config.diffusion_steps - 1),
            )
            alpha = self.alpha_bar[step].to(current.dtype)
            x0 = (
                current - (1.0 - alpha).sqrt() * predicted_noise
            ) / alpha.sqrt().clamp_min(1e-4)
            x0 = x0.clamp(-8.0, 8.0)
            if position + 1 == len(timesteps):
                current = x0
            else:
                previous = timesteps[position + 1]
                previous_alpha = self.alpha_bar[previous].to(current.dtype)
                current = previous_alpha.sqrt() * x0 + (
                    1.0 - previous_alpha
                ).sqrt() * predicted_noise
        samples = current.reshape(batch, ensemble, 20, 32, 64, 2)
        samples = samples * scale[:, None] + mean[:, None]
        return samples
