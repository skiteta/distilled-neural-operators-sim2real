from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .losses import (
    LossWeights,
    MetricAwareLoss,
    divergence_loss,
    spectral_relative_l1,
)
from .metrics import (
    error_to_score,
    profile_relative_l2,
    relative_l2,
    safe_prediction_score,
    score_batch,
    tke_relative_l2,
)
from .model import LatentFlowConfig, PhaseConditionedLatentFlow
from .train import (
    ExponentialMovingAverage,
    _forever,
    choose_device,
    load_checkpoint,
    make_dataset,
    make_loader,
    save_checkpoint,
    seed_everything,
)


def save_latent_flow_checkpoint(
    path: Path,
    model: PhaseConditionedLatentFlow,
    config: dict[str, Any],
    *,
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    torch.save(
        {
            "format_version": 2,
            "model_type": "phase_latent_flow",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": stage,
            "step": step,
            "metrics": metrics,
            "source_config": clean_config,
        },
        path,
    )


def load_latent_flow_checkpoint(
    path: str | Path, device: torch.device
) -> PhaseConditionedLatentFlow:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "phase_latent_flow":
        raise ValueError(f"not a phase_latent_flow checkpoint: {path}")
    model = PhaseConditionedLatentFlow(LatentFlowConfig.from_dict(checkpoint["model_config"])).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_latent_flow(
    config: dict[str, Any], device: torch.device
) -> PhaseConditionedLatentFlow:
    diffusion_config = config["diffusion"]
    teacher_path = Path(
        diffusion_config.get("teacher_checkpoint", config["train"]["real_checkpoint"])
    )
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    values = dict(diffusion_config)
    values["teacher"] = teacher_checkpoint["model_config"]
    model = PhaseConditionedLatentFlow(LatentFlowConfig.from_dict(values)).to(device)
    model.teacher.load_state_dict(teacher_checkpoint["state_dict"], strict=True)
    model.teacher.requires_grad_(False)
    model.teacher.eval()
    return model


@torch.inference_mode()
def validate_latent_flow(
    model: PhaseConditionedLatentFlow,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {
        "relative_l2": 0.0,
        "tke": 0.0,
        "profile": 0.0,
    }
    weighted_sps = 0.0
    weighted_coverage = 0.0
    scored_total = 0
    count = 0
    for count, (inputs, targets) in enumerate(loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction, half_width = model(inputs, return_interval=True)
        totals["relative_l2"] += float(relative_l2(prediction, targets))
        totals["tke"] += float(tke_relative_l2(prediction, targets))
        totals["profile"] += float(profile_relative_l2(prediction, targets))
        sps, coverage = safe_prediction_score(
            prediction,
            targets,
            lower=prediction - half_width,
            upper=prediction + half_width,
        )
        scored = int((targets[..., :2] != 0.0).count_nonzero())
        weighted_sps += float(sps) * scored
        weighted_coverage += float(coverage) * scored
        scored_total += scored
        if max_batches is not None and count >= max_batches:
            break
    if count == 0 or scored_total == 0:
        raise ValueError("latent-flow validation loader is empty")
    return {
        **{key: value / count for key, value in totals.items()},
        "sps_score": weighted_sps / scored_total,
        "coverage": weighted_coverage / scored_total,
    }


def train_latent_flow(config: dict[str, Any]) -> Path:
    seed_everything(int(config.get("seed", 20260808)) + 1)
    device = choose_device(config.get("device", "auto"))
    train_loader = make_loader(make_dataset(config, "real", "train"), config, shuffle=True)
    validation_loader = make_loader(
        make_dataset(config, "real", "validation"), config, shuffle=False
    )
    model = initialize_latent_flow(config, device)
    train_config = config["train"]
    diffusion_config = config["diffusion"]
    steps = int(diffusion_config.get("steps", 12000))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(diffusion_config.get("learning_rate", 2e-4)),
        weight_decay=float(diffusion_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_config.get("amp", True)) and device.type == "cuda"
    )
    ema = ExponentialMovingAverage(model, float(diffusion_config.get("ema_decay", 0.999)))
    batches = _forever(train_loader)
    output = Path(train_config["diffusion_checkpoint"])
    log_path = Path(train_config.get("checkpoint_dir", "runs")) / "latent_flow_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    throttle = max(0.0, float(train_config.get("throttle_seconds", 0.0)))

    weights = {
        "flow": float(diffusion_config.get("flow_weight", 1.0)),
        "phase": float(diffusion_config.get("phase_weight", 0.05)),
        "data": float(diffusion_config.get("data_weight", 0.15)),
        "tke": float(diffusion_config.get("tke_weight", 0.08)),
        "spectral": float(diffusion_config.get("spectral_weight", 0.04)),
        "divergence": float(diffusion_config.get("divergence_weight", 0.01)),
    }
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
                "spectral": spectral_relative_l1(prediction, targets),
                "divergence": divergence_loss(prediction),
            }
            loss = sum(weights[name] * value for name, value in components.items())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(diffusion_config.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle:
            time.sleep(throttle)

        if step % int(train_config.get("log_every", 50)) == 0:
            record = {
                "stage": "latent_flow",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in components.items()},
                "total": float(loss.detach()),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))

        val_every = int(diffusion_config.get("val_every", 500))
        if step % val_every == 0 or step == steps:
            metrics = validate_latent_flow(
                ema.shadow,
                validation_loader,
                device,
                max_batches=int(diffusion_config.get("val_batches", 16)),
            )
            point_quality = (
                float(error_to_score(metrics["relative_l2"]))
                + 0.2 * float(error_to_score(metrics["tke"]))
                + 0.2 * float(error_to_score(metrics["profile"]))
            )
            selection = point_quality + 0.6 * metrics["sps_score"]
            metrics["selection"] = selection
            print(
                json.dumps(
                    {"stage": "latent_flow", "step": step, "validation": metrics},
                    sort_keys=True,
                )
            )
            if selection > best:
                best = selection
                save_latent_flow_checkpoint(
                    output,
                    ema.shadow,
                    config,
                    stage="latent_flow",
                    step=step,
                    metrics=metrics,
                )
            model.train()
    return output


def refit_teacher_all_cases(config: dict[str, Any]) -> Path:
    """Refit the selected deterministic anchor on every official real case."""
    seed_everything(int(config.get("seed", 20260808)) + 2)
    device = choose_device(config.get("device", "auto"))
    diffusion_config = config["diffusion"]
    source = Path(diffusion_config["teacher_checkpoint"])
    model = load_checkpoint(source, device)
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    model.requires_grad_(True)
    model.width_head.requires_grad_(False)
    model.interval_temperature.requires_grad_(False)
    loader = make_loader(make_dataset(config, "real", "all"), config, shuffle=True)
    batches = _forever(loader)
    steps = int(diffusion_config.get("teacher_refit_steps", 2500))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(diffusion_config.get("teacher_refit_learning_rate", 2e-5)),
        weight_decay=float(diffusion_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config["train"].get("amp", True)) and device.type == "cuda"
    )
    objective = MetricAwareLoss(LossWeights.from_dict(config.get("loss", {}))).to(device)
    ema = ExponentialMovingAverage(model, float(diffusion_config.get("ema_decay", 0.999)))
    distillation_weight = float(diffusion_config.get("teacher_refit_distillation", 0.1))
    throttle = max(0.0, float(config["train"].get("throttle_seconds", 0.0)))
    log_every = int(config["train"].get("log_every", 50))
    log_path = (
        Path(config["train"].get("checkpoint_dir", "runs")) / "teacher_all82_refit_history.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_record: dict[str, float] = {}
    model.train()
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction = model(inputs)
            with torch.no_grad():
                reference_prediction = reference(inputs)
            metric_loss, components = objective(prediction, targets)
            distillation = relative_l2(prediction, reference_prediction)
            loss = metric_loss + distillation_weight * distillation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(diffusion_config.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle:
            time.sleep(throttle)
        if step % log_every == 0 or step == steps:
            last_record = {
                "stage": "teacher_all82_refit",
                "step": float(step),
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
                "distillation": float(distillation.detach()),
                "total_with_distillation": float(loss.detach()),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_record, sort_keys=True) + "\n")
            print(json.dumps(last_record, sort_keys=True))
    output = Path(config["train"]["teacher_refit_checkpoint"])
    save_checkpoint(
        output,
        ema.shadow,
        config,
        stage="teacher_all82_refit",
        step=steps,
        metrics=last_record,
    )
    return output


def refit_latent_flow_all_cases(config: dict[str, Any]) -> Path:
    """Adapt the selected probability flow to residuals from all official cases."""
    seed_everything(int(config.get("seed", 20260808)) + 3)
    device = choose_device(config.get("device", "auto"))
    source = Path(config["train"]["diffusion_checkpoint"])
    model = load_latent_flow_checkpoint(source, device)
    teacher_checkpoint = torch.load(
        config["train"]["teacher_refit_checkpoint"], map_location=device
    )
    model.teacher.load_state_dict(teacher_checkpoint["state_dict"], strict=True)
    model.teacher.requires_grad_(False)
    model.teacher.eval()
    loader = make_loader(make_dataset(config, "real", "all"), config, shuffle=True)
    batches = _forever(loader)
    diffusion_config = config["diffusion"]
    steps = int(diffusion_config.get("refit_steps", 3000))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(diffusion_config.get("refit_learning_rate", 5e-5)),
        weight_decay=float(diffusion_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config["train"].get("amp", True)) and device.type == "cuda"
    )
    ema = ExponentialMovingAverage(model, float(diffusion_config.get("ema_decay", 0.999)))
    weights = {
        "flow": float(diffusion_config.get("flow_weight", 1.0)),
        "phase": float(diffusion_config.get("phase_weight", 0.05)),
        "data": float(diffusion_config.get("data_weight", 0.15)),
        "tke": float(diffusion_config.get("tke_weight", 0.08)),
        "spectral": float(diffusion_config.get("spectral_weight", 0.04)),
        "divergence": float(diffusion_config.get("divergence_weight", 0.01)),
    }
    throttle = max(0.0, float(config["train"].get("throttle_seconds", 0.0)))
    log_every = int(config["train"].get("log_every", 50))
    log_path = (
        Path(config["train"].get("checkpoint_dir", "runs"))
        / "latent_flow_all82_refit_history.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_record: dict[str, float] = {}
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
                "spectral": spectral_relative_l1(prediction, targets),
                "divergence": divergence_loss(prediction),
            }
            loss = sum(weights[name] * value for name, value in components.items())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(diffusion_config.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle:
            time.sleep(throttle)
        if step % log_every == 0 or step == steps:
            last_record = {
                "stage": "latent_flow_all82_refit",
                "step": float(step),
                "learning_rate": scheduler.get_last_lr()[0],
                **{name: float(value.detach()) for name, value in components.items()},
                "total": float(loss.detach()),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_record, sort_keys=True) + "\n")
            print(json.dumps(last_record, sort_keys=True))
    output = Path(config["train"]["diffusion_refit_checkpoint"])
    save_latent_flow_checkpoint(
        output,
        ema.shadow,
        config,
        stage="latent_flow_all82_refit",
        step=steps,
        metrics=last_record,
    )
    return output


def finalize_latent_flow_refit(config: dict[str, Any]) -> Path:
    """Freeze the selected all-case refit with pre-declared inference settings.

    This deliberately does not optimize on the all-case training set. The point
    mode and interval scale must be selected beforehand and recorded in the YAML.
    """
    device = choose_device(config.get("device", "auto"))
    source = Path(config["train"]["diffusion_refit_checkpoint"])
    checkpoint = torch.load(source, map_location="cpu")
    model = load_latent_flow_checkpoint(source, device).eval()
    diffusion_config = config["diffusion"]
    point_mode = str(diffusion_config.get("final_point_mode", "anchor"))
    interval_scale = float(diffusion_config.get("final_interval_scale", 1.0))
    if point_mode not in {"anchor", "mean", "medoid", "mode"}:
        raise ValueError(f"unsupported final point mode: {point_mode}")
    if not interval_scale > 0.0:
        raise ValueError("final_interval_scale must be positive")
    model.config = replace(
        model.config,
        point_mode=point_mode,
        interval_scale=interval_scale,
    )
    target = Path(config["submission"]["checkpoint"])
    metrics = {
        "source_step": float(checkpoint.get("step", 0)),
        "interval_scale": interval_scale,
    }
    save_latent_flow_checkpoint(
        target,
        model,
        config,
        stage="latent_flow_all82_final",
        step=int(checkpoint.get("step", 0)),
        metrics=metrics,
    )
    print(
        json.dumps(
            {
                "finalized": str(target),
                "source": str(source),
                "point_mode": point_mode,
                "interval_scale": interval_scale,
            },
            sort_keys=True,
        )
    )
    return target


@torch.inference_mode()
def calibrate_latent_flow(config: dict[str, Any]) -> Path:
    device = choose_device(config.get("device", "auto"))
    source = Path(config["train"]["diffusion_checkpoint"])
    model = load_latent_flow_checkpoint(source, device).eval()
    loader = make_loader(make_dataset(config, "real", "validation"), config, shuffle=False)
    diffusion_config = config["diffusion"]
    scales = torch.logspace(
        torch.log10(torch.tensor(float(diffusion_config.get("interval_min", 0.25)))),
        torch.log10(torch.tensor(float(diffusion_config.get("interval_max", 4.0)))),
        steps=int(diffusion_config.get("interval_steps", 25)),
    ).tolist()
    point_modes = diffusion_config.get("calibration_point_modes", ["anchor", "mean", "medoid"])
    best: dict[str, Any] | None = None
    original = model.config
    for point_mode in point_modes:
        for scale in sorted({1.0, *(float(value) for value in scales)}):
            model.config = replace(original, point_mode=str(point_mode), interval_scale=scale)
            metrics = validate_latent_flow(
                model,
                loader,
                device,
                max_batches=int(diffusion_config.get("calibration_batches", 32)),
            )
            point_quality = (
                float(error_to_score(metrics["relative_l2"]))
                + 0.2 * float(error_to_score(metrics["tke"]))
                + 0.2 * float(error_to_score(metrics["profile"]))
            )
            selection = point_quality + 0.6 * metrics["sps_score"]
            candidate = {
                "point_mode": point_mode,
                "interval_scale": scale,
                "selection": selection,
                **metrics,
            }
            if best is None or candidate["selection"] > best["selection"]:
                best = candidate
    if best is None:
        raise RuntimeError("no latent-flow calibration candidate")
    model.config = replace(
        original,
        point_mode=str(best["point_mode"]),
        interval_scale=float(best["interval_scale"]),
    )
    target = Path(config["submission"]["checkpoint"])
    save_latent_flow_checkpoint(
        target,
        model,
        config,
        stage="latent_flow_calibrated",
        step=0,
        metrics={key: float(value) for key, value in best.items() if key != "point_mode"},
    )
    print(json.dumps({"calibrated": str(target), **best}, sort_keys=True))
    return target


@torch.inference_mode()
def evaluate_latent_flow(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config.get("device", "auto"))
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(config["train"]["diffusion_checkpoint"])
    model = load_latent_flow_checkpoint(checkpoint, device).eval()
    loader = make_loader(make_dataset(config, "real", "validation"), config, shuffle=False)
    metrics = validate_latent_flow(
        model,
        loader,
        device,
        max_batches=int(config["diffusion"].get("evaluation_batches", 64)),
    )
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    prediction, half_width = model(inputs, return_interval=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    public = score_batch(
        prediction,
        targets,
        elapsed_seconds=elapsed,
        lower=prediction - half_width,
        upper=prediction + half_width,
    )
    result = {
        "checkpoint": str(checkpoint),
        "holdout": metrics,
        "batch_seconds": elapsed,
        "first_batch_official_subscores": public.__dict__,
        "warning": ("The composite is inferred from public rows; Public transfer remains unknown."),
    }
    print(json.dumps(result, indent=2))
    return result


def clone_with_inference_settings(
    model: PhaseConditionedLatentFlow, **settings: Any
) -> PhaseConditionedLatentFlow:
    cloned = copy.deepcopy(model)
    cloned.config = replace(cloned.config, **settings)
    return cloned
