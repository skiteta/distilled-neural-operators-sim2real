from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import load_latent_flow_checkpoint
from .metrics import SIGMA_GLOBAL, safe_prediction_score
from .model import (
    DistilledIntervalConfig,
    DistilledUncertaintyOperator,
)
from .train import (
    _forever,
    choose_device,
    load_checkpoint,
    make_dataset,
    make_loader,
    seed_everything,
)


def save_distilled_checkpoint(
    path: str | Path,
    model: DistilledUncertaintyOperator,
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
            "format_version": 3,
            "model_type": "distilled_uncertainty",
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


def load_distilled_checkpoint(
    path: str | Path, device: torch.device
) -> DistilledUncertaintyOperator:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "distilled_uncertainty":
        raise ValueError(f"not a distilled_uncertainty checkpoint: {path}")
    model = DistilledUncertaintyOperator(
        DistilledIntervalConfig.from_dict(checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_distilled_model(
    config: dict[str, Any], device: torch.device
) -> DistilledUncertaintyOperator:
    values = config["distilled"]
    anchor_checkpoint = torch.load(values["anchor_checkpoint"], map_location=device)
    model_values = dict(values)
    model_values["anchor"] = anchor_checkpoint["model_config"]
    model = DistilledUncertaintyOperator(DistilledIntervalConfig.from_dict(model_values)).to(device)
    model.anchor.load_state_dict(anchor_checkpoint["state_dict"], strict=True)
    model.anchor.requires_grad_(False)
    model.anchor.eval()
    return model


def _smooth_velocity_map(field: torch.Tensor) -> torch.Tensor:
    channels_first = field.permute(0, 4, 1, 2, 3)
    smoothed = F.avg_pool3d(
        channels_first,
        kernel_size=(3, 5, 5),
        stride=1,
        padding=(1, 2, 2),
    )
    return smoothed.permute(0, 2, 3, 4, 1)


def distilled_interval_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    half_width: torch.Tensor,
    diffusion_width: torch.Tensor,
    sim_prediction: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train an SPS-aware interval using complementary uncertainty teachers."""
    values = config["distilled"]
    pred = prediction[..., :2]
    true = target[..., :2]
    width = half_width[..., :2].clamp_min(1e-7)
    residual = (pred - true).abs()
    mask = true != 0.0
    if not bool(mask.any()):
        raise ValueError("training batch has no measured velocity element")

    quantile = float(values.get("quantile", 0.90))
    delta = residual - width
    pinball_element = torch.maximum(quantile * delta, (quantile - 1.0) * delta)
    pinball = pinball_element[mask].mean() / SIGMA_GLOBAL

    # The probability-flow spread represents phase/aleatoric ambiguity. The
    # low-pass sim-vs-real disagreement represents cross-regime epistemic risk.
    epistemic = _smooth_velocity_map((sim_prediction[..., :2] - pred).abs())
    aleatoric = _smooth_velocity_map(diffusion_width[..., :2])
    epistemic_weight = float(values.get("epistemic_weight", 0.50))
    teacher_width = torch.sqrt(aleatoric.square() + epistemic_weight * epistemic.square() + 1e-10)
    input_std = prediction.new_tensor(float(values.get("teacher_width_max", 0.50)))
    teacher_width = teacher_width.clamp(max=input_std)
    log_width = torch.log1p(width / SIGMA_GLOBAL)
    log_teacher = torch.log1p(teacher_width / SIGMA_GLOBAL)
    distillation = F.smooth_l1_loss(log_width[mask], log_teacher[mask])

    # Differentiable proxy for the official inside-interval indicator multiplied
    # by its exact exponential sharpness term exp(-2*half_width/sigma_global).
    softness = SIGMA_GLOBAL * float(values.get("inside_softness", 0.10))
    inside_probability = torch.sigmoid((width - residual) / max(softness, 1e-6))
    safe_reward = inside_probability * torch.exp(-2.0 * width / SIGMA_GLOBAL)
    sps_surrogate = -safe_reward[mask].mean()

    loss = (
        float(values.get("pinball_weight", 1.0)) * pinball
        + float(values.get("distillation_weight", 0.35)) * distillation
        + float(values.get("sps_surrogate_weight", 0.20)) * sps_surrogate
    )
    return loss, {
        "loss": float(loss.detach()),
        "pinball": float(pinball.detach()),
        "distillation": float(distillation.detach()),
        "sps_surrogate": float(sps_surrogate.detach()),
        "mean_half_width": float(width[mask].mean().detach()),
        "mean_aleatoric_teacher": float(aleatoric[mask].mean().detach()),
        "mean_epistemic_teacher": float(epistemic[mask].mean().detach()),
        "mean_combined_teacher": float(teacher_width[mask].mean().detach()),
    }


@torch.inference_mode()
def validate_distilled(
    model: DistilledUncertaintyOperator,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 2,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Report global and per-regime SPS; selection penalizes the worst regime."""
    model.eval()
    model.interval_temperature.fill_(float(temperature))
    indices_by_regime: dict[tuple[float, float], list[int]] = {}
    for index, (case_index, _) in enumerate(dataset.windows):
        regime = dataset.cases[case_index].regime
        indices_by_regime.setdefault(regime, []).append(index)

    case_scores: dict[str, float] = {}
    case_coverages: dict[str, float] = {}
    total_weighted_sps = 0.0
    total_weighted_coverage = 0.0
    total_scored = 0
    for regime, indices in indices_by_regime.items():
        weighted_sps = 0.0
        weighted_coverage = 0.0
        case_scored = 0
        for start in range(0, len(indices), batch_size):
            samples = [dataset[index] for index in indices[start : start + batch_size]]
            inputs = torch.stack([sample[0] for sample in samples]).to(device)
            targets = torch.stack([sample[1] for sample in samples]).to(device)
            prediction, half_width = model(inputs, return_interval=True)
            score, coverage = safe_prediction_score(
                prediction,
                targets,
                lower=prediction - half_width,
                upper=prediction + half_width,
            )
            scored = int((targets[..., :2] != 0.0).count_nonzero())
            weighted_sps += float(score) * scored
            weighted_coverage += float(coverage) * scored
            case_scored += scored
        key = f"Re{regime[0]:g}_AoA{regime[1]:g}"
        case_scores[key] = weighted_sps / case_scored
        case_coverages[key] = weighted_coverage / case_scored
        total_weighted_sps += weighted_sps
        total_weighted_coverage += weighted_coverage
        total_scored += case_scored

    scores = list(case_scores.values())
    global_sps = total_weighted_sps / total_scored
    median_sps = statistics.median(scores)
    worst_sps = min(scores)
    # Hidden Re/AoA shift makes mean-only selection brittle. This criterion keeps
    # most weight on official aggregate SPS while explicitly rewarding robustness.
    selection = 0.55 * global_sps + 0.30 * median_sps + 0.15 * worst_sps
    return {
        "sps_score": global_sps,
        "coverage": total_weighted_coverage / total_scored,
        "case_median_sps": median_sps,
        "case_worst_sps": worst_sps,
        "selection": selection,
        "temperature": float(temperature),
        "case_sps": case_scores,
        "case_coverage": case_coverages,
    }


def train_distilled_uncertainty(config: dict[str, Any]) -> Path:
    seed_everything(int(config.get("seed", 20260810)))
    device = choose_device(config.get("device", "auto"))
    values = config["distilled"]
    train_dataset = make_dataset(config, "real", "train")
    validation_dataset = make_dataset(config, "real", "validation")
    train_loader = make_loader(
        train_dataset,
        config,
        shuffle=True,
        batch_size=int(values.get("batch_size", 1)),
    )
    batches = _forever(train_loader)

    student = initialize_distilled_model(config, device)
    diffusion = load_latent_flow_checkpoint(values["diffusion_checkpoint"], device).eval()
    diffusion.requires_grad_(False)
    sim_model = load_checkpoint(values["sim_checkpoint"], device).eval()
    sim_model.requires_grad_(False)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    steps = int(values.get("steps", 1200))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(values.get("learning_rate", 4e-4)),
        weight_decay=float(values.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    )
    output = Path(values["checkpoint"])
    history = Path(values.get("history", "runs/distilled_uncertainty/history.jsonl"))
    history.parent.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    throttle = max(0.0, float(values.get("throttle_seconds", 0.05)))
    log_every = int(values.get("log_every", 25))
    val_every = int(values.get("val_every", 100))
    student.train()
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _, diffusion_width = diffusion(inputs, return_interval=True)
            sim_prediction = sim_model(inputs)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction, half_width = student(inputs, return_interval=True)
            loss, components = distilled_interval_loss(
                prediction,
                targets,
                half_width,
                diffusion_width,
                sim_prediction,
                config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, float(values.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if throttle:
            time.sleep(throttle)

        if step % log_every == 0:
            record = {
                "stage": "distilled_uncertainty",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
            }
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % val_every == 0 or step == steps:
            metrics = validate_distilled(
                student,
                validation_dataset,
                device,
                batch_size=int(values.get("validation_batch_size", 2)),
            )
            print(
                json.dumps(
                    {"stage": "distilled_uncertainty", "step": step, "validation": metrics},
                    sort_keys=True,
                ),
                flush=True,
            )
            if metrics["selection"] > best:
                best = float(metrics["selection"])
                save_distilled_checkpoint(
                    output,
                    student,
                    config,
                    stage="distilled_uncertainty",
                    step=step,
                    metrics=metrics,
                )
            student.train()
    return output


@torch.inference_mode()
def calibrate_distilled_uncertainty(config: dict[str, Any]) -> Path:
    device = choose_device(config.get("device", "auto"))
    values = config["distilled"]
    source = Path(values["checkpoint"])
    checkpoint = torch.load(source, map_location="cpu")
    model = load_distilled_checkpoint(source, device).eval()
    validation = make_dataset(config, "real", "validation")
    temperatures = torch.logspace(
        torch.log10(torch.tensor(float(values.get("temperature_min", 0.50)))),
        torch.log10(torch.tensor(float(values.get("temperature_max", 2.50)))),
        int(values.get("temperature_steps", 17)),
    ).tolist()
    best: dict[str, Any] | None = None
    for temperature in sorted({1.0, *(float(value) for value in temperatures)}):
        metrics = validate_distilled(
            model,
            validation,
            device,
            batch_size=int(values.get("validation_batch_size", 2)),
            temperature=temperature,
        )
        if best is None or metrics["selection"] > best["selection"]:
            best = metrics
    if best is None:
        raise RuntimeError("no temperature candidate was evaluated")
    model.interval_temperature.fill_(float(best["temperature"]))
    output = Path(config["submission"]["checkpoint"])
    save_distilled_checkpoint(
        output,
        model,
        config,
        stage="distilled_uncertainty_calibrated",
        step=int(checkpoint.get("step", 0)),
        metrics=best,
    )
    print(json.dumps({"calibrated": str(output), **best}, sort_keys=True))
    return output


@torch.inference_mode()
def evaluate_distilled_uncertainty(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config.get("device", "auto"))
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(config["distilled"]["checkpoint"])
    model = load_distilled_checkpoint(checkpoint, device).eval()
    metrics = validate_distilled(
        model,
        make_dataset(config, "real", "validation"),
        device,
        batch_size=int(config["distilled"].get("validation_batch_size", 2)),
        temperature=float(model.interval_temperature),
    )
    result = {"checkpoint": str(checkpoint), "holdout": metrics}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
