from __future__ import annotations

import copy
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .losses import divergence_loss, spectral_relative_l1
from .metrics import (
    SIGMA_GLOBAL,
    error_to_score,
    inferred_composite_score,
    profile_relative_l2,
    relative_l2,
    safe_prediction_score,
    tke_relative_l2,
)
from .model import MaskedSpectralPosteriorRefiner, SpectralPosteriorConfig
from .public_proxy import LocalScores, PublicTransferModel
from .train import (
    ExponentialMovingAverage,
    _forever,
    choose_device,
    make_dataset,
    make_loader,
    seed_everything,
)


def save_spectral_posterior_checkpoint(
    path: str | Path,
    model: MaskedSpectralPosteriorRefiner,
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
            "format_version": 4,
            "model_type": "masked_spectral_posterior",
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


def load_spectral_posterior_checkpoint(
    path: str | Path, device: torch.device
) -> MaskedSpectralPosteriorRefiner:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "masked_spectral_posterior":
        raise ValueError(f"not a masked_spectral_posterior checkpoint: {path}")
    model = MaskedSpectralPosteriorRefiner(
        SpectralPosteriorConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_spectral_posterior(
    config: dict[str, Any], device: torch.device
) -> MaskedSpectralPosteriorRefiner:
    values = dict(config["spectral_posterior"])
    anchor_checkpoint = torch.load(values["anchor_checkpoint"], map_location=device)
    values["anchor"] = anchor_checkpoint["model_config"]
    interval_checkpoint = None
    if values.get("interval_teacher_checkpoint"):
        interval_checkpoint = torch.load(values["interval_teacher_checkpoint"], map_location=device)
        values["interval_teacher"] = interval_checkpoint["model_config"]
    model = MaskedSpectralPosteriorRefiner(SpectralPosteriorConfig.from_dict(values)).to(device)
    model.anchor.load_state_dict(anchor_checkpoint["state_dict"], strict=True)
    model.anchor.requires_grad_(False)
    model.anchor.eval()
    if interval_checkpoint is not None and model.interval_teacher is not None:
        model.interval_teacher.load_state_dict(interval_checkpoint["state_dict"], strict=True)
        model.interval_teacher.requires_grad_(False)
        model.interval_teacher.eval()
    return model


def load_spectral_posterior_for_config(
    path: str | Path, config: dict[str, Any], device: torch.device
) -> MaskedSpectralPosteriorRefiner:
    """Load a checkpoint and attach a newly configured frozen interval teacher."""
    model = load_spectral_posterior_checkpoint(path, device)
    teacher_path = config["spectral_posterior"].get("interval_teacher_checkpoint")
    if not teacher_path or model.interval_teacher is not None:
        return model
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    upgraded_config = replace(
        model.config,
        interval_teacher=teacher_checkpoint["model_config"],
    )
    upgraded = MaskedSpectralPosteriorRefiner(upgraded_config).to(device)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("interval_teacher.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected interval-teacher upgrade keys: {incompatible}")
    if upgraded.interval_teacher is None:
        raise RuntimeError("interval teacher was not constructed")
    upgraded.interval_teacher.load_state_dict(teacher_checkpoint["state_dict"], strict=True)
    upgraded.interval_teacher.requires_grad_(False)
    upgraded.interval_teacher.eval()
    return upgraded


@torch.inference_mode()
def estimate_spectral_scale(
    model: MaskedSpectralPosteriorRefiner,
    dataset,
    device: torch.device,
    *,
    max_samples: int,
    batch_size: int,
    minimum_ratio: float,
) -> dict[str, float]:
    """Whiten residual coefficients using real-train statistics only."""
    model.eval()
    count = min(len(dataset), int(max_samples))
    if count <= 0:
        raise ValueError("spectral scale requires at least one real training sample")
    if count == len(dataset):
        indices = list(range(count))
    else:
        indices = torch.linspace(0, len(dataset) - 1, count).round().long().tolist()

    squared: torch.Tensor | None = None
    seen = 0
    for start in range(0, len(indices), batch_size):
        samples = [dataset[index] for index in indices[start : start + batch_size]]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        targets = torch.stack([sample[1] for sample in samples]).to(device)
        anchor = model.anchor(inputs)
        _, std = model._statistics(inputs)
        residual = model._target_residual(anchor, targets)
        packed = model.pack_fourier(residual / std)
        batch_squared = packed.square().sum(dim=0, keepdim=True)
        squared = batch_squared if squared is None else squared + batch_squared
        seen += len(samples)
    if squared is None:
        raise RuntimeError("spectral scale accumulation produced no coefficients")
    scale = (squared / seen).clamp_min(0.0).sqrt()
    model.set_spectral_scale(scale, minimum_ratio=minimum_ratio)
    positive = model.spectral_scale[model.spectral_scale > 0]
    return {
        "samples": float(seen),
        "minimum": float(positive.min()),
        "median": float(positive.median()),
        "maximum": float(positive.max()),
    }


def _three_channel(velocity: torch.Tensor) -> torch.Tensor:
    return torch.cat((velocity, torch.zeros_like(velocity[..., :1])), dim=-1)


def temporal_phase_alignment_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Amplitude-independent temporal Fourier phase/coherence objective."""
    pred = prediction[..., :2].float()
    true = target[..., :2].float()
    pred = pred - pred.mean(dim=1, keepdim=True)
    true = true - true.mean(dim=1, keepdim=True)
    pred_spectrum = torch.fft.rfft(pred, dim=1, norm="ortho")
    true_spectrum = torch.fft.rfft(true, dim=1, norm="ortho")
    dimensions = tuple(range(1, pred_spectrum.ndim))
    coherence = (pred_spectrum.conj() * true_spectrum).real.sum(dim=dimensions)
    denominator = (
        pred_spectrum.abs().square().sum(dim=dimensions).sqrt()
        * true_spectrum.abs().square().sum(dim=dimensions).sqrt()
    ).clamp_min(1e-8)
    return (1.0 - coherence / denominator).mean()


def temporal_increment_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return relative_l2(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])


@torch.inference_mode()
def validate_spectral_posterior(
    model: MaskedSpectralPosteriorRefiner,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    runtime_score: float = 85.0,
) -> dict[str, float]:
    """Use one concatenated tensor, matching the submission-equivalent scorer."""
    model.eval()
    predictions = []
    targets_all = []
    widths = []
    for count, (inputs, targets) in enumerate(loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction, half_width = model(inputs, return_interval=True)
        predictions.append(prediction)
        targets_all.append(targets)
        widths.append(half_width)
        if max_batches is not None and count >= max_batches:
            break
    if not predictions:
        raise ValueError("spectral-posterior validation loader is empty")
    prediction = torch.cat(predictions)
    target = torch.cat(targets_all)
    half_width = torch.cat(widths)
    rel = relative_l2(prediction, target)
    tke = tke_relative_l2(prediction, target)
    profile = profile_relative_l2(prediction, target)
    sps, coverage = safe_prediction_score(
        prediction,
        target,
        lower=prediction - half_width,
        upper=prediction + half_width,
    )
    rel_score = error_to_score(rel)
    tke_score = error_to_score(tke)
    profile_score = error_to_score(profile)
    final = inferred_composite_score(rel_score, tke_score, profile_score, runtime_score, sps)
    return {
        "relative_l2": float(rel),
        "tke": float(tke),
        "profile": float(profile),
        "rel_l2_score": float(rel_score),
        "tke_score": float(tke_score),
        "mvpe_score": float(profile_score),
        "sps_score": float(sps),
        "coverage": float(coverage),
        "inferred_final_score": float(final),
        "selection": float(final),
    }


def train_spectral_posterior(config: dict[str, Any], stage: str) -> Path:
    if stage not in {"sim", "real"}:
        raise ValueError("stage must be sim or real")
    seed_everything(int(config.get("seed", 20260810)) + (0 if stage == "sim" else 1))
    device = choose_device(config.get("device", "auto"))
    values = config["spectral_posterior"]
    if stage == "sim":
        model = initialize_spectral_posterior(config, device)
        scale_metrics = estimate_spectral_scale(
            model,
            make_dataset(config, "real", "train"),
            device,
            max_samples=int(values.get("scale_samples", 128)),
            batch_size=int(values.get("scale_batch_size", 2)),
            minimum_ratio=float(values.get("scale_floor_ratio", 0.05)),
        )
        print(json.dumps({"spectral_scale": scale_metrics}, sort_keys=True), flush=True)
    else:
        model = load_spectral_posterior_for_config(values["sim_checkpoint"], config, device)

    train_loader = make_loader(
        make_dataset(config, "sim" if stage == "sim" else "real", "train"),
        config,
        shuffle=True,
        batch_size=int(values.get("batch_size", 1)),
    )
    validation_loader = make_loader(
        make_dataset(config, "real", "validation"),
        config,
        shuffle=False,
        batch_size=int(values.get("validation_batch_size", 2)),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    steps = int(values[f"{stage}_steps"])
    learning_rate = float(values[f"{stage}_learning_rate"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(values.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(values.get("minimum_learning_rate", 1e-6))
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    )
    ema = ExponentialMovingAverage(model, float(values.get("ema_decay", 0.999)))
    batches = _forever(train_loader)
    weights = {
        name: float(values.get(f"{name}_weight", default))
        for name, default in {
            "flow": 1.0,
            "endpoint": 0.15,
            "data": 0.20,
            "tke": 0.10,
            "profile": 0.03,
            "spectral": 0.05,
            "divergence": 0.01,
        }.items()
    }
    output = Path(values[f"{stage}_checkpoint"])
    history = Path(values.get("history_dir", "runs/track1_spectral_posterior"))
    history.mkdir(parents=True, exist_ok=True)
    history_file = history / f"{stage}_history.jsonl"
    best = float("-inf")
    throttle = max(0.0, float(values.get("throttle_seconds", 0.05)))
    log_every = int(values.get("log_every", 25))
    val_every = int(values.get("val_every", 250))
    runtime_score = float(values.get("runtime_score_assumption", 85.0))

    model.train()
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction, latent_losses = model.flow_matching_prediction(inputs, targets)
            components = {
                **latent_losses,
                "data": relative_l2(prediction, targets),
                "tke": tke_relative_l2(prediction, targets),
                "profile": profile_relative_l2(prediction, targets),
                "spectral": spectral_relative_l1(prediction, targets),
                "divergence": divergence_loss(prediction),
            }
            loss = sum(weights[name] * value for name, value in components.items())
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
                "stage": f"spectral_posterior_{stage}",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in components.items()},
                "total": float(loss.detach()),
            }
            with history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % val_every == 0 or step == steps:
            metrics = validate_spectral_posterior(
                ema.shadow,
                validation_loader,
                device,
                max_batches=int(values.get("val_batches", 8)),
                runtime_score=runtime_score,
            )
            print(
                json.dumps(
                    {"stage": f"spectral_posterior_{stage}", "step": step, "validation": metrics},
                    sort_keys=True,
                ),
                flush=True,
            )
            if metrics["selection"] > best:
                best = metrics["selection"]
                save_spectral_posterior_checkpoint(
                    output,
                    ema.shadow,
                    config,
                    stage=f"spectral_posterior_{stage}",
                    step=step,
                    metrics=metrics,
                )
            model.train()
    return output


def _spectral_interval_adaptation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    half_width: torch.Tensor,
    reference_width: torch.Tensor,
    values: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = prediction[..., :2]
    true = target[..., :2]
    width = half_width[..., :2].clamp_min(1e-7)
    reference = reference_width[..., :2].clamp_min(1e-7)
    residual = (pred - true).abs()
    mask = true != 0.0
    quantile = float(values.get("interval_quantile", 0.90))
    delta = residual - width
    pinball = torch.maximum(quantile * delta, (quantile - 1.0) * delta)[mask].mean()
    pinball = pinball / SIGMA_GLOBAL
    distillation = F.smooth_l1_loss(
        torch.log1p(width[mask] / SIGMA_GLOBAL),
        torch.log1p(reference[mask] / SIGMA_GLOBAL),
    )
    softness = SIGMA_GLOBAL * float(values.get("interval_inside_softness", 0.10))
    inside_probability = torch.sigmoid((width - residual) / max(softness, 1e-6))
    safe_reward = inside_probability * torch.exp(-2.0 * width / SIGMA_GLOBAL)
    sps_surrogate = -safe_reward[mask].mean()
    loss = (
        float(values.get("interval_pinball_weight", 1.0)) * pinball
        + float(values.get("interval_distillation_weight", 0.20)) * distillation
        + float(values.get("interval_sps_weight", 0.25)) * sps_surrogate
    )
    return loss, {
        "loss": float(loss.detach()),
        "pinball": float(pinball.detach()),
        "distillation": float(distillation.detach()),
        "sps_surrogate": float(sps_surrogate.detach()),
        "mean_half_width": float(width[mask].mean().detach()),
    }


def train_spectral_interval(config: dict[str, Any]) -> Path:
    """Adapt only the frozen teacher's interval head to the spectral point forecast."""
    seed_everything(int(config.get("seed", 20260810)) + 2)
    device = choose_device(config.get("device", "auto"))
    values = config["spectral_posterior"]
    interval_source = values.get("interval_source_checkpoint", values["real_checkpoint"])
    model = load_spectral_posterior_for_config(interval_source, config, device)
    model.config = replace(
        model.config,
        sample_steps=int(values.get("sample_steps", model.config.sample_steps)),
        correction_source=str(values.get("interval_point_source", "mode")),
        point_correction_scale=float(
            values.get("interval_point_scale", model.config.point_correction_scale)
        ),
        interval_scale=0.0,
        anchor_width_floor=1.0,
    )
    if model.interval_teacher is None:
        raise ValueError("interval_teacher_checkpoint is required for interval adaptation")
    reference = copy.deepcopy(model.interval_teacher).eval()
    reference.requires_grad_(False)
    model.requires_grad_(False)
    for name, parameter in model.interval_teacher.named_parameters():
        if not name.startswith("anchor.") and name != "interval_temperature":
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no interval-head parameter was enabled")

    train_loader = make_loader(
        make_dataset(config, "real", "train"),
        config,
        shuffle=True,
        batch_size=int(values.get("interval_batch_size", 1)),
    )
    validation_loader = make_loader(
        make_dataset(config, "real", "validation"),
        config,
        shuffle=False,
        batch_size=int(values.get("validation_batch_size", 2)),
    )
    batches = _forever(train_loader)
    steps = int(values.get("interval_adaptation_steps", 1000))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(values.get("interval_learning_rate", 1e-4)),
        weight_decay=float(values.get("interval_weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    )
    output = Path(values["interval_checkpoint"])
    history = Path(values.get("history_dir", "runs/track1_spectral_posterior"))
    history.mkdir(parents=True, exist_ok=True)
    history_file = history / "interval_history.jsonl"
    runtime_score = float(values.get("runtime_score_assumption", 85.0))
    baseline = validate_spectral_posterior(
        model,
        validation_loader,
        device,
        max_batches=int(values.get("val_batches", 105)),
        runtime_score=runtime_score,
    )
    best = baseline["selection"]
    save_spectral_posterior_checkpoint(
        output,
        model,
        config,
        stage="spectral_interval_baseline",
        step=0,
        metrics=baseline,
    )
    print(json.dumps({"stage": "spectral_interval", "step": 0, "validation": baseline}))

    throttle = max(0.0, float(values.get("throttle_seconds", 0.05)))
    log_every = int(values.get("interval_log_every", 25))
    val_every = int(values.get("interval_val_every", 100))
    model.eval()
    model.interval_teacher.train()
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction = model(inputs)
            _, reference_width = reference(inputs, return_interval=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _, half_width = model.interval_teacher(inputs, return_interval=True)
            loss, components = _spectral_interval_adaptation_loss(
                prediction, targets, half_width, reference_width, values
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, float(values.get("interval_grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if throttle:
            time.sleep(throttle)

        if step % log_every == 0 or step == 1:
            record = {
                "stage": "spectral_interval",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
            }
            with history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % val_every == 0 or step == steps:
            metrics = validate_spectral_posterior(
                model,
                validation_loader,
                device,
                max_batches=int(values.get("val_batches", 105)),
                runtime_score=runtime_score,
            )
            print(
                json.dumps(
                    {"stage": "spectral_interval", "step": step, "validation": metrics},
                    sort_keys=True,
                ),
                flush=True,
            )
            if metrics["selection"] > best:
                best = metrics["selection"]
                save_spectral_posterior_checkpoint(
                    output,
                    model,
                    config,
                    stage="spectral_interval",
                    step=step,
                    metrics=metrics,
                )
            model.eval()
            model.interval_teacher.train()
    return output


@torch.inference_mode()
def _cache_posterior_components(model, dataset, device, batch_size: int) -> dict[str, Any]:
    model.eval()
    cached: dict[str, list[torch.Tensor]] = {
        "anchor": [],
        "mean": [],
        "mode": [],
        "spread": [],
        "anchor_width": [],
        "target": [],
    }
    regimes: list[tuple[float, float]] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        samples = [dataset[index] for index in indices]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        targets = torch.stack([sample[1] for sample in samples]).to(device)
        anchor, mean, mode, spread, anchor_width = model.posterior_components(inputs)
        cached["anchor"].append(anchor[..., :2])
        cached["mean"].append(mean)
        cached["mode"].append(mode)
        cached["spread"].append(spread)
        cached["anchor_width"].append(anchor_width[..., :2])
        cached["target"].append(targets)
        regimes.extend(dataset.cases[dataset.windows[index][0]].regime for index in indices)
    result: dict[str, Any] = {name: torch.cat(values) for name, values in cached.items()}
    result["regimes"] = regimes
    return result


def _score_candidate(
    point_uv: torch.Tensor,
    width_uv: torch.Tensor,
    target: torch.Tensor,
    runtime_score: float,
) -> dict[str, float]:
    prediction = _three_channel(point_uv)
    half_width = _three_channel(width_uv)
    rel = relative_l2(prediction, target)
    tke = tke_relative_l2(prediction, target)
    profile = profile_relative_l2(prediction, target)
    sps, coverage = safe_prediction_score(
        prediction,
        target,
        lower=prediction - half_width,
        upper=prediction + half_width,
    )
    rel_score = error_to_score(rel)
    tke_score = error_to_score(tke)
    profile_score = error_to_score(profile)
    final = inferred_composite_score(rel_score, tke_score, profile_score, runtime_score, sps)
    return {
        "relative_l2": float(rel),
        "tke": float(tke),
        "profile": float(profile),
        "rel_l2_score": float(rel_score),
        "tke_score": float(tke_score),
        "mvpe_score": float(profile_score),
        "sps_score": float(sps),
        "coverage": float(coverage),
        "inferred_final_score": float(final),
    }


@torch.inference_mode()
def _cache_mode_components(
    model: MaskedSpectralPosteriorRefiner,
    dataset,
    device: torch.device,
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    """Cache the exact deployed mode path once for joint alpha/interval search."""
    model.eval()
    cached: dict[str, list[torch.Tensor]] = {
        "anchor": [],
        "correction": [],
        "anchor_width": [],
        "target": [],
    }
    regimes: list[tuple[float, float]] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        samples = [dataset[index] for index in indices]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        targets = torch.stack([sample[1] for sample in samples]).to(device)
        _, correction, anchor = model.mode_prediction(
            inputs, correction_scale=1.0, steps=steps
        )
        if model.interval_teacher is not None:
            _, anchor_width = model.interval_teacher(inputs, return_interval=True)
        else:
            _, anchor_width = model.anchor(inputs, return_interval=True)
        cached["anchor"].append(anchor[..., :2])
        cached["correction"].append(correction)
        cached["anchor_width"].append(anchor_width[..., :2])
        cached["target"].append(targets)
        regimes.extend(dataset.cases[dataset.windows[index][0]].regime for index in indices)
    if not cached["anchor"]:
        raise ValueError("mode validation dataset is empty")
    result: dict[str, Any] = {name: torch.cat(parts) for name, parts in cached.items()}
    result["regimes"] = regimes
    return result


def _public_proxy_final(
    transfer: PublicTransferModel | None,
    metrics: dict[str, float],
    *,
    runtime_score: float,
    runtime_family: str,
    point_fingerprint: str | None,
) -> tuple[float, dict[str, Any] | None]:
    if transfer is None:
        return float(metrics["inferred_final_score"]), None
    proxy = transfer.estimate(
        LocalScores(
            relative_l2=float(metrics["relative_l2"]),
            tke=float(metrics["tke"]),
            mvpe=float(metrics["profile"]),
            time_score=float(runtime_score),
            sps_score=float(metrics["sps_score"]),
        ),
        runtime_family=runtime_family,
        point_fingerprint=point_fingerprint,
    )
    return float(proxy["scores"]["final_score"]), proxy


@torch.inference_mode()
def evaluate_spectral_mode_candidates(
    model: MaskedSpectralPosteriorRefiner,
    dataset,
    device: torch.device,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Select a mode correction only when it survives full OOF and TKE guards."""
    steps = int(values.get("mode_sample_steps", 1))
    cached = _cache_mode_components(
        model,
        dataset,
        device,
        batch_size=int(values.get("validation_batch_size", 2)),
        steps=steps,
    )
    alphas = sorted(
        {
            0.0,
            *(
                float(value)
                for value in values.get(
                    "mode_validation_alphas",
                    [0.0, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375],
                )
            ),
        }
    )
    floors = sorted(
        {
            float(value)
            for value in values.get("mode_validation_width_floors", [0.9, 1.0, 1.1])
        }
    )
    runtime_score = float(
        values.get("mode_runtime_score_assumption", values.get("runtime_score_assumption", 85.0))
    )
    runtime_family = str(values.get("mode_runtime_family", "one_pass"))
    transfer_path = Path(values.get("public_transfer_config", "configs/public_transfer_v9.json"))
    transfer = PublicTransferModel.load(transfer_path) if transfer_path.exists() else None
    unique_regimes = sorted(set(cached["regimes"]))
    candidates: list[dict[str, Any]] = []

    for alpha in alphas:
        point = model.corrected_point(cached["anchor"], cached["correction"], alpha)
        for floor in floors:
            width = cached["anchor_width"] * floor
            metrics = _score_candidate(point, width, cached["target"], runtime_score)
            case_finals: dict[str, float] = {}
            case_tke: dict[str, float] = {}
            for regime in unique_regimes:
                indices = [
                    index for index, observed in enumerate(cached["regimes"]) if observed == regime
                ]
                selected = torch.tensor(indices, device=device)
                case_metrics = _score_candidate(
                    point.index_select(0, selected),
                    width.index_select(0, selected),
                    cached["target"].index_select(0, selected),
                    runtime_score,
                )
                name = f"Re{regime[0]:g}_AoA{regime[1]:g}"
                case_finals[name] = case_metrics["inferred_final_score"]
                case_tke[name] = case_metrics["tke"]
            finals = list(case_finals.values())
            robust_local = (
                0.70 * metrics["inferred_final_score"]
                + 0.20 * statistics.median(finals)
                + 0.10 * min(finals)
            )
            point_fingerprint = (
                str(values.get("anchor_point_fingerprint"))
                if alpha == 0.0 and values.get("anchor_point_fingerprint")
                else None
            )
            proxy_final, proxy = _public_proxy_final(
                transfer,
                metrics,
                runtime_score=runtime_score,
                runtime_family=runtime_family,
                point_fingerprint=point_fingerprint,
            )
            candidates.append(
                {
                    "point_correction_scale": alpha,
                    "anchor_width_floor": floor,
                    "interval_scale": 0.0,
                    "correction_source": "mode",
                    **metrics,
                    "case_median_final": statistics.median(finals),
                    "case_worst_final": min(finals),
                    "case_final": case_finals,
                    "case_tke": case_tke,
                    "robust_local_score": robust_local,
                    "public_proxy_final": proxy_final,
                    "public_proxy": proxy,
                }
            )

    baseline = max(
        (item for item in candidates if item["point_correction_scale"] == 0.0),
        key=lambda item: 0.8 * item["public_proxy_final"] + 0.2 * item["robust_local_score"],
    )
    tke_limit = baseline["tke"] + float(values.get("mode_tke_tolerance", 0.001))
    local_limit = baseline["inferred_final_score"] - float(
        values.get("mode_local_score_tolerance", 0.02)
    )
    for candidate in candidates:
        candidate["tke_delta_from_anchor"] = candidate["tke"] - baseline["tke"]
        candidate["local_delta_from_anchor"] = (
            candidate["inferred_final_score"] - baseline["inferred_final_score"]
        )
        candidate["eligible"] = bool(
            candidate["tke"] <= tke_limit
            and candidate["inferred_final_score"] >= local_limit
        )
        candidate["mode_selection"] = (
            0.80 * candidate["public_proxy_final"]
            + 0.20 * candidate["robust_local_score"]
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("mode validation rejected every candidate including the anchor")
    best = max(eligible, key=lambda item: item["mode_selection"])
    nonzero = [
        candidate
        for candidate in eligible
        if float(candidate["point_correction_scale"]) > 0.0
    ]
    best_nonzero = max(nonzero, key=lambda item: item["mode_selection"]) if nonzero else None
    ordered = sorted(candidates, key=lambda item: item["mode_selection"], reverse=True)
    return {
        "sample_count": len(cached["regimes"]),
        "mode_sample_steps": steps,
        "baseline": baseline,
        "best": best,
        "best_nonzero": best_nonzero,
        "top_candidates": ordered[: int(values.get("mode_report_top", 12))],
    }


def train_spectral_mode(config: dict[str, Any]) -> Path:
    """Fine-tune the exact zero-initialized one-step mode with physics metrics."""
    seed_everything(int(config.get("seed", 20260810)) + 3)
    device = choose_device(config.get("device", "auto"))
    values = config["spectral_posterior"]
    source = values.get("mode_source_checkpoint", values["real_checkpoint"])
    model = load_spectral_posterior_for_config(source, config, device)
    mode_steps = int(values.get("mode_sample_steps", 1))
    train_scale = float(values.get("mode_train_scale", 0.125))
    model.config = replace(
        model.config,
        sample_steps=mode_steps,
        correction_source="mode",
        point_correction_scale=train_scale,
        interval_scale=0.0,
        anchor_width_floor=1.0,
        preserve_anchor_tke=bool(
            values.get("mode_preserve_anchor_tke", model.config.preserve_anchor_tke)
        ),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no spectral mode parameter is trainable")
    train_loader = make_loader(
        make_dataset(config, "real", "train"),
        config,
        shuffle=True,
        batch_size=int(values.get("mode_batch_size", 1)),
    )
    validation_dataset = make_dataset(config, "real", "validation")
    batches = _forever(train_loader)
    steps = int(values.get("mode_steps", 3000))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(values.get("mode_learning_rate", 2e-5)),
        weight_decay=float(values.get("mode_weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(values.get("mode_minimum_learning_rate", 5e-7)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    )
    ema = ExponentialMovingAverage(model, float(values.get("mode_ema_decay", 0.995)))
    output = Path(values["mode_checkpoint"])
    candidate_dir = Path(values.get("mode_candidates_dir", "checkpoints/spectral_mode_candidates"))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(values.get("history_dir", "runs/track1_spectral_posterior"))
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "mode_history.jsonl"
    best_selection = float("-inf")

    def validate_and_save(step: int) -> dict[str, Any]:
        nonlocal best_selection
        report = evaluate_spectral_mode_candidates(
            ema.shadow, validation_dataset, device, values
        )
        chosen = report["best"]
        ema.shadow.config = replace(
            ema.shadow.config,
            sample_steps=mode_steps,
            correction_source="mode",
            point_correction_scale=float(chosen["point_correction_scale"]),
            interval_scale=0.0,
            anchor_width_floor=float(chosen["anchor_width_floor"]),
        )
        snapshot = candidate_dir / f"step_{step:04d}.pt"
        save_spectral_posterior_checkpoint(
            snapshot,
            ema.shadow,
            config,
            stage="spectral_mode_candidate",
            step=step,
            metrics=report,
        )
        if float(chosen["mode_selection"]) > best_selection:
            best_selection = float(chosen["mode_selection"])
            save_spectral_posterior_checkpoint(
                output,
                ema.shadow,
                config,
                stage="spectral_mode_refinement",
                step=step,
                metrics=report,
            )
        print(
            json.dumps(
                {"stage": "spectral_mode", "step": step, "validation": report},
                sort_keys=True,
            ),
            flush=True,
        )
        return report

    validate_and_save(0)
    model.train()
    throttle = max(0.0, float(values.get("mode_throttle_seconds", 0.02)))
    log_every = int(values.get("mode_log_every", 25))
    val_every = int(values.get("mode_val_every", 250))
    weights = {
        "data": float(values.get("mode_data_weight", 1.0)),
        "tke": float(values.get("mode_tke_weight", 1.5)),
        "tke_guard": float(values.get("mode_tke_guard_weight", 2.0)),
        "profile": float(values.get("mode_profile_weight", 0.2)),
        "phase": float(values.get("mode_phase_weight", 0.0)),
        "temporal": float(values.get("mode_temporal_weight", 0.0)),
        "spectral": float(values.get("mode_spectral_weight", 0.1)),
        "divergence": float(values.get("mode_divergence_weight", 0.01)),
        "flow": float(values.get("mode_flow_weight", 0.2)),
        "endpoint": float(values.get("mode_endpoint_weight", 0.03)),
    }
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction, correction, anchor = model.mode_prediction(
                inputs, correction_scale=train_scale, steps=mode_steps
            )
            tke = tke_relative_l2(prediction, targets)
            with torch.no_grad():
                anchor_tke = tke_relative_l2(anchor, targets)
            _, latent_losses = model.flow_matching_prediction(inputs, targets)
            components = {
                "data": relative_l2(prediction, targets),
                "tke": tke,
                "tke_guard": F.relu(tke - anchor_tke),
                "profile": profile_relative_l2(prediction, targets),
                "phase": temporal_phase_alignment_loss(prediction, targets),
                "temporal": temporal_increment_relative_l2(prediction, targets),
                "spectral": spectral_relative_l1(prediction, targets),
                "divergence": divergence_loss(prediction),
                "flow": latent_losses["flow"],
                "endpoint": latent_losses["endpoint"],
            }
            loss = sum(weights[name] * value for name, value in components.items())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, float(values.get("mode_grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle:
            time.sleep(throttle)

        if step % log_every == 0 or step == 1:
            record = {
                "stage": "spectral_mode",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in components.items()},
                "anchor_tke": float(anchor_tke.detach()),
                "correction_rms": float(correction.square().mean().sqrt().detach()),
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
def calibrate_spectral_posterior(config: dict[str, Any]) -> Path:
    """Jointly tune posterior point correction and SPS interval on held-out regimes."""
    device = choose_device(config.get("device", "auto"))
    values = config["spectral_posterior"]
    source = Path(values.get("calibration_checkpoint", values["real_checkpoint"]))
    checkpoint = torch.load(source, map_location="cpu")
    model = load_spectral_posterior_for_config(source, config, device).eval()
    model.config = replace(
        model.config,
        sample_steps=int(values.get("sample_steps", model.config.sample_steps)),
    )
    dataset = make_dataset(config, "real", "validation")
    cached = _cache_posterior_components(
        model,
        dataset,
        device,
        int(values.get("validation_batch_size", 2)),
    )
    scales = torch.logspace(
        torch.log10(torch.tensor(float(values.get("interval_min", 0.25)))),
        torch.log10(torch.tensor(float(values.get("interval_max", 4.0)))),
        int(values.get("interval_steps", 13)),
    ).tolist()
    scales = sorted({0.0, 1.0, *(float(value) for value in scales)})
    alphas = sorted(
        {float(value) for value in values.get("correction_scales", [0, 0.125, 0.25, 0.5])}
    )
    sources = [str(value) for value in values.get("correction_sources", ["mean", "mode"])]
    floors = sorted({float(value) for value in values.get("anchor_width_floors", [0.0, 0.25])})
    runtime_score = float(values.get("runtime_score_assumption", 85.0))

    candidates: list[dict[str, Any]] = []
    seen_points: set[tuple[str, float]] = set()
    for source_name in sources:
        if source_name not in {"mean", "mode"}:
            raise ValueError(f"unknown correction source: {source_name}")
        correction = cached[source_name]
        for alpha in alphas:
            point_key = ("anchor" if alpha == 0.0 else source_name, alpha)
            if point_key in seen_points:
                continue
            seen_points.add(point_key)
            point = cached["anchor"] + alpha * correction
            for floor in floors:
                for scale in scales:
                    width = cached["spread"] * scale
                    if floor > 0.0:
                        width = torch.maximum(width, cached["anchor_width"] * floor)
                    metrics = _score_candidate(point, width, cached["target"], runtime_score)
                    candidates.append(
                        {
                            "correction_source": source_name,
                            "point_correction_scale": alpha,
                            "interval_scale": scale,
                            "anchor_width_floor": floor,
                            **metrics,
                        }
                    )

    if not candidates:
        raise RuntimeError("no spectral posterior calibration candidates")
    candidates.sort(key=lambda item: item["inferred_final_score"], reverse=True)

    # Re-rank the strongest aggregate candidates across complete Reynolds/AoA
    # regimes. This discourages a narrow interval that wins only on one case.
    top_count = min(len(candidates), int(values.get("robust_top_candidates", 12)))
    unique_regimes = sorted(set(cached["regimes"]))
    for candidate in candidates[:top_count]:
        source_name = str(candidate["correction_source"])
        alpha = float(candidate["point_correction_scale"])
        point = cached["anchor"] + alpha * cached[source_name]
        width = cached["spread"] * float(candidate["interval_scale"])
        floor = float(candidate["anchor_width_floor"])
        if floor > 0.0:
            width = torch.maximum(width, cached["anchor_width"] * floor)
        case_finals: dict[str, float] = {}
        case_sps: dict[str, float] = {}
        for regime in unique_regimes:
            indices = [index for index, value in enumerate(cached["regimes"]) if value == regime]
            selected = torch.tensor(indices, device=device)
            metrics = _score_candidate(
                point.index_select(0, selected),
                width.index_select(0, selected),
                cached["target"].index_select(0, selected),
                runtime_score,
            )
            name = f"Re{regime[0]:g}_AoA{regime[1]:g}"
            case_finals[name] = metrics["inferred_final_score"]
            case_sps[name] = metrics["sps_score"]
        finals = list(case_finals.values())
        candidate["case_median_final"] = statistics.median(finals)
        candidate["case_worst_final"] = min(finals)
        candidate["case_final"] = case_finals
        candidate["case_sps"] = case_sps
        candidate["robust_selection"] = (
            0.70 * candidate["inferred_final_score"]
            + 0.20 * candidate["case_median_final"]
            + 0.10 * candidate["case_worst_final"]
        )
    best = max(
        candidates[:top_count],
        key=lambda item: item.get("robust_selection", float("-inf")),
    )
    model.config = replace(
        model.config,
        correction_source=str(best["correction_source"]),
        point_correction_scale=float(best["point_correction_scale"]),
        interval_scale=float(best["interval_scale"]),
        anchor_width_floor=float(best["anchor_width_floor"]),
    )
    output = Path(config["submission"]["checkpoint"])
    save_spectral_posterior_checkpoint(
        output,
        model,
        config,
        stage="spectral_posterior_calibrated",
        step=int(checkpoint.get("step", 0)),
        metrics=best,
    )
    print(json.dumps({"calibrated": str(output), **best}, sort_keys=True), flush=True)
    return output


@torch.inference_mode()
def evaluate_spectral_posterior(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config.get("device", "auto"))
    values = config["spectral_posterior"]
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(values["real_checkpoint"])
    model = load_spectral_posterior_for_config(checkpoint, config, device).eval()
    loader = make_loader(
        make_dataset(config, "real", "validation"),
        config,
        shuffle=False,
        batch_size=int(values.get("validation_batch_size", 2)),
    )
    metrics = validate_spectral_posterior(
        model,
        loader,
        device,
        max_batches=int(values.get("evaluation_batches", 105)),
        runtime_score=float(values.get("runtime_score_assumption", 85.0)),
    )
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    model(inputs, return_interval=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "checkpoint": str(checkpoint),
        "holdout": metrics,
        "batch_seconds": elapsed,
        "seconds_per_sample": elapsed / len(inputs),
        "note": "Final acceptance must use evaluate_submission_equivalent.py on all 210 windows.",
    }
    print(json.dumps(result, indent=2), flush=True)
    return result
