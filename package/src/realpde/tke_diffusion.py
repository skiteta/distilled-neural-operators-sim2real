from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import relative_l2, turbulent_kinetic_energy
from .model import MaskedSpectralPosteriorRefiner, OperatorBlock, SpectralPosteriorConfig
from .public_proxy import LocalScores, PublicTransferModel
from .spectral_posterior import _score_candidate
from .train import (
    ExponentialMovingAverage,
    _forever,
    choose_device,
    make_dataset,
    make_loader,
    seed_everything,
)


@dataclass(frozen=True)
class TKEDiffusionConfig:
    """Conditional probability flow for the future spatial TKE residual."""

    base: dict[str, Any]
    width: int = 64
    condition_depth: int = 3
    denoiser_depth: int = 6
    spectral_rank: int = 16
    modes_y: int = 12
    modes_x: int = 24
    expansion: int = 2
    dropout: float = 0.02
    frequency_bins: int = 10
    time_embedding_channels: int = 16
    target_clip: float = 4.0
    sample_steps: int = 4
    ensemble_size: int = 4
    max_ensemble_size: int = 4
    sample_seed: int = 20260810
    correction_scale: float = 1.0
    maximum_log_amplitude: float = 0.4
    interval_width_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TKEDiffusionConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in dict(values).items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrequencyConditionedTKEDiffusion(nn.Module):
    """Generate a future TKE map from temporal Fourier phase and base dynamics."""

    map_feature_channels = 10

    def __init__(self, config: TKEDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.base = MaskedSpectralPosteriorRefiner(
            SpectralPosteriorConfig.from_dict(config.base)
        )
        self.base.requires_grad_(False)
        self.base.eval()
        bins = config.frequency_bins
        self.condition_channels = 8 * bins + self.map_feature_channels + 6
        self.condition_lift = nn.Conv2d(self.condition_channels, config.width, 1)
        self.condition_blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.condition_depth)
            ]
        )
        self.denoiser_lift = nn.Conv2d(
            config.width + 1 + config.time_embedding_channels, config.width, 1
        )
        self.denoiser_blocks = nn.ModuleList(
            [
                OperatorBlock(
                    config.width,
                    config.spectral_rank,
                    config.modes_y,
                    config.modes_x,
                    config.expansion,
                    config.dropout,
                )
                for _ in range(config.denoiser_depth)
            ]
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.velocity_head = nn.Sequential(
            nn.GroupNorm(groups, config.width),
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.width, 1, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        generator = torch.Generator(device="cpu").manual_seed(config.sample_seed)
        half = (config.max_ensemble_size + 1) // 2
        noise = torch.randn(half, 1, 32, 64, generator=generator)
        antithetic = torch.stack((noise, -noise), dim=1).flatten(0, 1)
        antithetic = antithetic[: config.max_ensemble_size]
        self.register_buffer("noise_bank", antithetic, persistent=True)
        y = torch.linspace(-1.0, 1.0, 32)
        x = torch.linspace(-1.0, 1.0, 64)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer(
            "spatial_coordinates", torch.stack((grid_y, grid_x), dim=0), persistent=True
        )

    def train(self, mode: bool = True) -> FrequencyConditionedTKEDiffusion:
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def temporal_tke(flow: torch.Tensor) -> torch.Tensor:
        return turbulent_kinetic_energy(flow[..., :2].float())

    def _map_features(
        self, observed: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        full = self.temporal_tke(observed)
        first = self.temporal_tke(observed[:, :10])
        recent = self.temporal_tke(observed[:, 10:])
        quarters = [
            self.temporal_tke(observed[:, index : index + 5])
            for index in (0, 5, 10, 15)
        ]
        base = self.temporal_tke(base_prediction)
        maps = torch.stack(
            (base, full, first, recent, *quarters, base - full, recent - first), dim=1
        )
        return torch.sign(maps) * torch.log1p(maps.abs() / 1e-3)

    def _frequency_features(
        self, flow: torch.Tensor, normalization: torch.Tensor
    ) -> torch.Tensor:
        uv = flow[..., :2].float()
        centered = uv - uv.mean(dim=1, keepdim=True)
        normalized = centered / normalization
        spectrum = torch.fft.rfft(normalized, dim=1, norm="ortho")
        bins = min(self.config.frequency_bins, spectrum.shape[1] - 1)
        selected = spectrum[:, 1 : bins + 1]
        if bins < self.config.frequency_bins:
            padding = selected.new_zeros(
                selected.shape[0],
                self.config.frequency_bins - bins,
                *selected.shape[2:],
            )
            selected = torch.cat((selected, padding), dim=1)
        packed = torch.view_as_real(selected).permute(0, 1, 4, 5, 2, 3)
        return packed.reshape(flow.shape[0], -1, 32, 64)

    def condition(
        self, observed: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        observed_uv = observed[..., :2].float()
        component_scale = observed_uv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        observed_frequency = self._frequency_features(observed, component_scale)
        base_frequency = self._frequency_features(base_prediction, component_scale)
        map_features = self._map_features(observed, base_prediction)
        observed_mean = observed_uv.mean(dim=1).permute(0, 3, 1, 2) / component_scale[
            :, 0, 0, 0, :, None, None
        ]
        base_uv = base_prediction[..., :2].float()
        base_mean = base_uv.mean(dim=1).permute(0, 3, 1, 2) / component_scale[
            :, 0, 0, 0, :, None, None
        ]
        coordinates = self.spatial_coordinates[None].expand(observed.shape[0], -1, -1, -1)
        features = torch.cat(
            (
                observed_frequency,
                base_frequency,
                map_features,
                observed_mean,
                base_mean,
                coordinates,
            ),
            dim=1,
        ).to(observed.dtype)
        if features.shape[1] != self.condition_channels:
            raise RuntimeError(f"unexpected TKE diffusion condition: {features.shape}")
        hidden = self.condition_lift(features)
        for block in self.condition_blocks:
            hidden = block(hidden)
        return hidden

    def _time_embedding(self, time_value: torch.Tensor) -> torch.Tensor:
        half = self.config.time_embedding_channels // 2
        frequency = 2.0 ** torch.arange(
            half, device=time_value.device, dtype=time_value.dtype
        )
        phase = torch.pi * time_value[:, None] * frequency[None]
        embedded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        if embedded.shape[1] < self.config.time_embedding_channels:
            embedded = F.pad(
                embedded, (0, self.config.time_embedding_channels - embedded.shape[1])
            )
        return embedded[:, :, None, None].expand(-1, -1, 32, 64)

    def velocity(
        self, latent: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.denoiser_lift(
            torch.cat((latent, condition, self._time_embedding(time_value)), dim=1)
        )
        for block in self.denoiser_blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden)

    def target_latent(
        self, base_prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_tke = self.temporal_tke(base_prediction)
        true_tke = self.temporal_tke(target)
        reference = base_tke.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        latent = ((true_tke - base_tke) / reference).clamp(
            -float(self.config.target_clip), float(self.config.target_clip)
        )
        return latent[:, None], reference, true_tke

    @staticmethod
    def reconstruct_tke(
        base_prediction: torch.Tensor, latent: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        base_tke = FrequencyConditionedTKEDiffusion.temporal_tke(base_prediction)
        return (base_tke + latent[:, 0] * reference).clamp_min(0.0)

    @staticmethod
    def reproject_tke(
        base_prediction: torch.Tensor,
        predicted_tke: torch.Tensor,
        correction_scale: float,
        maximum_log_amplitude: float,
    ) -> torch.Tensor:
        uv = base_prediction[..., :2].float()
        mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - mean
        base_tke = FrequencyConditionedTKEDiffusion.temporal_tke(uv)
        target_tke = base_tke + float(correction_scale) * (predicted_tke - base_tke)
        log_amplitude = 0.5 * torch.log(
            target_tke.clamp_min(1e-10) / base_tke.clamp_min(1e-10)
        )
        maximum = float(maximum_log_amplitude)
        factor = log_amplitude.clamp(-maximum, maximum).exp()
        corrected = mean + fluctuation * factor[:, None, :, :, None]
        return torch.cat((corrected, torch.zeros_like(corrected[..., :1])), dim=-1)

    def flow_matching_prediction(
        self, observed: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            base_prediction = self.base(observed)
        condition = self.condition(observed, base_prediction)
        target_latent, reference, true_tke = self.target_latent(base_prediction, target)
        noise = torch.randn_like(target_latent)
        time_value = torch.rand(
            target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype
        )
        shaped_time = time_value[:, None, None, None]
        interpolated = (1.0 - shaped_time) * noise + shaped_time * target_latent
        velocity = self.velocity(interpolated, time_value, condition)
        velocity_target = target_latent - noise
        endpoint = interpolated + (1.0 - shaped_time) * velocity
        predicted_tke = self.reconstruct_tke(base_prediction, endpoint, reference)
        map_dimensions = (1, 2)
        map_error = torch.linalg.vector_norm(predicted_tke - true_tke, dim=map_dimensions)
        map_norm = torch.linalg.vector_norm(true_tke, dim=map_dimensions).clamp_min(1e-8)
        point = self.reproject_tke(
            base_prediction,
            predicted_tke,
            1.0,
            float(self.config.maximum_log_amplitude),
        )
        return predicted_tke, {
            "flow": (velocity - velocity_target).square().mean(),
            "endpoint": F.smooth_l1_loss(endpoint, target_latent),
            "map": (map_error / map_norm).mean(),
            "data": relative_l2(point, target),
        }

    def _integrate(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        steps = int(self.config.sample_steps)
        if steps < 1:
            raise ValueError("TKE diffusion sample_steps must be positive")
        result = latent
        step_size = 1.0 / steps
        for index in range(steps):
            time_value = result.new_full((result.shape[0],), index / steps)
            result = result + step_size * self.velocity(result, time_value, condition)
        return result

    def components(
        self, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base_prediction, half_width = self.base(observed, return_interval=True)
        condition = self.condition(observed, base_prediction)
        ensemble = int(self.config.ensemble_size)
        if not 1 <= ensemble <= int(self.config.max_ensemble_size):
            raise ValueError("TKE diffusion ensemble_size exceeds noise bank")
        batch = observed.shape[0]
        repeated_condition = condition[:, None].expand(-1, ensemble, -1, -1, -1)
        repeated_condition = repeated_condition.reshape(
            batch * ensemble, condition.shape[1], 32, 64
        )
        initial = self.noise_bank[:ensemble][None].expand(batch, -1, -1, -1, -1)
        initial = initial.reshape(batch * ensemble, 1, 32, 64).to(condition.dtype)
        sampled = self._integrate(initial, repeated_condition)
        mean_latent = sampled.reshape(batch, ensemble, 1, 32, 64).mean(dim=1)
        base_tke = self.temporal_tke(base_prediction)
        reference = base_tke.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        predicted_tke = self.reconstruct_tke(base_prediction, mean_latent, reference)
        return base_prediction, half_width, predicted_tke

    def forward(
        self, observed: torch.Tensor, *, return_interval: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base, half_width, predicted_tke = self.components(observed)
        point = self.reproject_tke(
            base,
            predicted_tke,
            self.config.correction_scale,
            self.config.maximum_log_amplitude,
        )
        half_width = half_width * float(self.config.interval_width_scale)
        if return_interval:
            return point, half_width
        return point


def save_tke_diffusion_checkpoint(
    path: str | Path,
    model: FrequencyConditionedTKEDiffusion,
    config: dict[str, Any],
    *,
    stage: str,
    step: int,
    metrics: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    torch.save(
        {
            "format_version": 6,
            "model_type": "frequency_conditioned_tke_diffusion",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": stage,
            "step": int(step),
            "metrics": metrics,
            "source_config": clean_config,
        },
        output,
    )
    return output


def load_tke_diffusion_checkpoint(
    path: str | Path, device: torch.device
) -> FrequencyConditionedTKEDiffusion:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "frequency_conditioned_tke_diffusion":
        raise ValueError(f"not a frequency-conditioned TKE checkpoint: {path}")
    model = FrequencyConditionedTKEDiffusion(
        TKEDiffusionConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_tke_diffusion(
    config: dict[str, Any], device: torch.device
) -> FrequencyConditionedTKEDiffusion:
    values = dict(config["tke_diffusion"])
    base_checkpoint = torch.load(values["base_checkpoint"], map_location=device)
    if base_checkpoint.get("model_type") != "masked_spectral_posterior":
        raise ValueError("TKE diffusion base must be a masked spectral posterior")
    values["base"] = base_checkpoint["model_config"]
    model = FrequencyConditionedTKEDiffusion(TKEDiffusionConfig.from_dict(values)).to(device)
    model.base.load_state_dict(base_checkpoint["state_dict"], strict=True)
    model.base.requires_grad_(False)
    model.base.eval()
    return model


@torch.inference_mode()
def _cache_components(
    model: FrequencyConditionedTKEDiffusion,
    dataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    cached: dict[str, list[torch.Tensor]] = {
        "base": [],
        "width": [],
        "predicted_tke": [],
        "target": [],
    }
    regimes: list[tuple[float, float]] = []
    model.eval()
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        samples = [dataset[index] for index in indices]
        observed = torch.stack([sample[0] for sample in samples]).to(device)
        target = torch.stack([sample[1] for sample in samples]).to(device)
        base, width, predicted_tke = model.components(observed)
        cached["base"].append(base)
        cached["width"].append(width)
        cached["predicted_tke"].append(predicted_tke)
        cached["target"].append(target)
        regimes.extend(dataset.cases[dataset.windows[index][0]].regime for index in indices)
    result: dict[str, Any] = {name: torch.cat(parts) for name, parts in cached.items()}
    result["regimes"] = regimes
    return result


@torch.inference_mode()
def evaluate_tke_diffusion_candidates(
    model: FrequencyConditionedTKEDiffusion,
    dataset,
    device: torch.device,
    values: dict[str, Any],
) -> dict[str, Any]:
    cached = _cache_components(
        model, dataset, device, int(values.get("validation_batch_size", 2))
    )
    scales = sorted(
        {0.0, *(float(value) for value in values.get("correction_scales", [0, 0.25, 0.5, 1]))}
    )
    caps = sorted(
        {float(value) for value in values.get("maximum_log_amplitudes", [0.05, 0.1, 0.2, 0.4])}
    )
    width_scales = sorted(
        {float(value) for value in values.get("interval_width_scales", [1.0])}
    )
    runtime_score = float(values.get("runtime_score_assumption", 90.0))
    transfer = PublicTransferModel.load(
        values.get("public_transfer_config", "configs/public_transfer_v9.json")
    )
    unique_regimes = sorted(set(cached["regimes"]))
    true_tke = model.temporal_tke(cached["target"])
    map_numerator = torch.linalg.vector_norm(
        (cached["predicted_tke"] - true_tke).flatten(1), dim=1
    )
    map_denominator = torch.linalg.vector_norm(true_tke.flatten(1), dim=1).clamp_min(1e-8)
    predicted_tke_relative_l2 = float((map_numerator / map_denominator).mean())
    candidates: list[dict[str, Any]] = []
    for scale in scales:
        for cap in caps:
            point = model.reproject_tke(
                cached["base"], cached["predicted_tke"], scale, cap
            )
            for width_scale in width_scales:
                width = cached["width"] * width_scale
                metrics = _score_candidate(
                    point[..., :2], width[..., :2], cached["target"], runtime_score
                )
                case_finals: dict[str, float] = {}
                case_tke: dict[str, float] = {}
                for regime in unique_regimes:
                    indices = torch.tensor(
                        [
                            index
                            for index, observed in enumerate(cached["regimes"])
                            if observed == regime
                        ],
                        device=device,
                    )
                    case = _score_candidate(
                        point.index_select(0, indices)[..., :2],
                        width.index_select(0, indices)[..., :2],
                        cached["target"].index_select(0, indices),
                        runtime_score,
                    )
                    name = f"Re{regime[0]:g}_AoA{regime[1]:g}"
                    case_finals[name] = case["inferred_final_score"]
                    case_tke[name] = case["tke"]
                finals = list(case_finals.values())
                robust = (
                    0.70 * metrics["inferred_final_score"]
                    + 0.20 * statistics.median(finals)
                    + 0.10 * min(finals)
                )
                proxy = transfer.estimate(
                    LocalScores(
                        relative_l2=metrics["relative_l2"],
                        tke=metrics["tke"],
                        mvpe=metrics["profile"],
                        time_score=runtime_score,
                        sps_score=metrics["sps_score"],
                    ),
                    runtime_family="one_pass",
                )
                proxy_final = float(proxy["scores"]["final_score"])
                candidates.append(
                    {
                        "correction_scale": scale,
                        "maximum_log_amplitude": cap,
                        "interval_width_scale": width_scale,
                        **metrics,
                        "case_final": case_finals,
                        "case_tke": case_tke,
                        "case_worst_final": min(finals),
                        "robust_local_score": robust,
                        "public_proxy": proxy,
                        "public_proxy_final": proxy_final,
                        "selection": 0.8 * proxy_final + 0.2 * robust,
                    }
                )
    baseline = max(
        (item for item in candidates if item["correction_scale"] == 0.0),
        key=lambda item: item["selection"],
    )
    tolerance = float(values.get("local_score_tolerance", 0.03))
    eligible = [
        item
        for item in candidates
        if item["inferred_final_score"] >= baseline["inferred_final_score"] - tolerance
    ]
    best = max(eligible, key=lambda item: item["selection"])
    ordered = sorted(candidates, key=lambda item: item["selection"], reverse=True)
    return {
        "sample_count": len(cached["regimes"]),
        "predicted_tke_relative_l2": predicted_tke_relative_l2,
        "baseline": baseline,
        "best": best,
        "top_candidates": ordered[: int(values.get("report_top", 12))],
    }


def train_tke_diffusion(config: dict[str, Any], stage: str) -> Path:
    if stage not in {"sim", "real"}:
        raise ValueError("TKE diffusion stage must be sim or real")
    seed_everything(int(config.get("seed", 20260810)) + (6 if stage == "sim" else 7))
    device = choose_device(config.get("device", "auto"))
    values = config["tke_diffusion"]
    if stage == "sim":
        model = initialize_tke_diffusion(config, device)
    else:
        source = Path(values["sim_checkpoint"])
        model = (
            load_tke_diffusion_checkpoint(source, device)
            if source.exists()
            else initialize_tke_diffusion(config, device)
        )
        model.config = replace(
            model.config,
            correction_scale=float(values.get("correction_scale", 1.0)),
            maximum_log_amplitude=float(values.get("maximum_log_amplitude", 0.4)),
            interval_width_scale=float(values.get("interval_width_scale", 1.0)),
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loader = make_loader(
        make_dataset(config, "sim" if stage == "sim" else "real", "train"),
        config,
        shuffle=True,
        batch_size=int(values.get("batch_size", 2)),
    )
    validation_dataset = make_dataset(config, "real", "validation")
    batches = _forever(loader)
    steps = int(values[f"{stage}_steps"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(values[f"{stage}_learning_rate"]),
        weight_decay=float(values.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(values.get("minimum_learning_rate", 1e-6)),
    )
    amp_requested = bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    has_complex_parameters = any(parameter.is_complex() for parameter in parameters)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_requested and not has_complex_parameters)
    ema = ExponentialMovingAverage(model, float(values.get("ema_decay", 0.995)))
    output = Path(values[f"{stage}_checkpoint"])
    candidate_dir = Path(values.get("candidates_dir", "checkpoints/tke_diffusion_candidates"))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(values.get("history_dir", "runs/track1_tke_diffusion"))
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{stage}_history.jsonl"
    best_selection = float("-inf")

    if amp_requested and has_complex_parameters:
        print(
            json.dumps(
                {
                    "stage": f"tke_diffusion_{stage}",
                    "amp_enabled": False,
                    "reason": "complex spectral parameters require FP32",
                }
            ),
            flush=True,
        )

    def validate_and_save(step: int) -> None:
        nonlocal best_selection
        report = evaluate_tke_diffusion_candidates(
            ema.shadow, validation_dataset, device, values
        )
        chosen = report["best"]
        ema.shadow.config = replace(
            ema.shadow.config,
            correction_scale=float(chosen["correction_scale"]),
            maximum_log_amplitude=float(chosen["maximum_log_amplitude"]),
            interval_width_scale=float(chosen["interval_width_scale"]),
        )
        if stage == "real":
            save_tke_diffusion_checkpoint(
                candidate_dir / f"step_{step:04d}.pt",
                ema.shadow,
                config,
                stage="tke_diffusion_candidate",
                step=step,
                metrics=report,
            )
        if float(chosen["selection"]) > best_selection:
            best_selection = float(chosen["selection"])
            save_tke_diffusion_checkpoint(
                output,
                ema.shadow,
                config,
                stage=f"tke_diffusion_{stage}",
                step=step,
                metrics=report,
            )
        print(
            json.dumps(
                {"stage": f"tke_diffusion_{stage}", "step": step, "validation": report}
            ),
            flush=True,
        )

    validate_and_save(0)
    weights = {
        "flow": float(values.get("flow_weight", 1.0)),
        "endpoint": float(values.get("endpoint_weight", 0.5)),
        "map": float(values.get("map_weight", 1.0)),
    }
    model.train()
    log_every = int(values.get("log_every", 25))
    val_every = int(values.get("val_every", 500))
    throttle = max(0.0, float(values.get("throttle_seconds", 0.02)))
    for step in range(1, steps + 1):
        observed, target = next(batches)
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _, components = model.flow_matching_prediction(observed, target)
            loss = sum(weights[name] * components[name] for name in weights)
            loss = loss + float(values.get("data_weight", 0.1)) * components["data"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, float(values.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle:
            time.sleep(throttle)
        if step % log_every == 0 or step == 1:
            record = {
                "stage": f"tke_diffusion_{stage}",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in components.items()},
                "total": float(loss.detach()),
            }
            with history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % val_every == 0 or step == steps:
            validate_and_save(step)
            model.train()
    return output


@torch.inference_mode()
def evaluate_tke_diffusion(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config.get("device", "auto"))
    values = config["tke_diffusion"]
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(values["real_checkpoint"])
    model = load_tke_diffusion_checkpoint(checkpoint, device).eval()
    report = evaluate_tke_diffusion_candidates(
        model, make_dataset(config, "real", "validation"), device, values
    )
    result = {"checkpoint": str(checkpoint), "validation": report}
    print(json.dumps(result, indent=2), flush=True)
    return result
