from __future__ import annotations

import copy
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FlowWindowDataset, discover_cases
from .losses import LossWeights, MetricAwareLoss, interval_quantile_loss
from .metrics import (
    profile_relative_l2,
    relative_l2,
    safe_prediction_score,
    tke_relative_l2,
)
from .model import HybridFlowOperator, ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_dataset(config: dict[str, Any], source: str, split: str) -> FlowWindowDataset:
    data = config["data"]
    cases = discover_cases(
        data[f"{source}_dir"],
        exclude_files=(data.get("exclude_files", []) if source == "real" else []),
    )
    return FlowWindowDataset(
        cases,
        in_steps=data.get("in_steps", 20),
        out_steps=data.get("out_steps", 20),
        stride=data.get("stride", 20),
        spatial_stride=data.get("spatial_stride", 2),
        split=split,
        holdout_regimes=data.get("holdout_regimes", []),
    )


def make_loader(
    dataset: FlowWindowDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    workers = int(config["data"].get("num_workers", 0))
    configured_batch = int(config["train"].get("batch_size", 8))
    return DataLoader(
        dataset,
        batch_size=batch_size or configured_batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= configured_batch,
    )


def _forever(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name, parameter in self.shadow.named_parameters():
            parameter.lerp_(source[name], 1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.shadow.named_buffers():
            buffer.copy_(source_buffers[name])


@torch.inference_mode()
def validate(
    model: HybridFlowOperator,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"relative_l2": 0.0, "tke": 0.0, "profile": 0.0}
    count = 0
    for count, (inputs, targets) in enumerate(loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs)
        totals["relative_l2"] += float(relative_l2(predictions, targets))
        totals["tke"] += float(tke_relative_l2(predictions, targets))
        totals["profile"] += float(profile_relative_l2(predictions, targets))
        if max_batches is not None and count >= max_batches:
            break
    if count == 0:
        raise ValueError("validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def checkpoint_path(config: dict[str, Any], stage: str) -> Path:
    return Path(config["train"][f"{stage}_checkpoint"])


def save_checkpoint(
    path: Path,
    model: HybridFlowOperator,
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
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "stage": stage,
            "step": step,
            "metrics": metrics,
            "source_config": clean_config,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> HybridFlowOperator:
    checkpoint = torch.load(path, map_location=device)
    model = HybridFlowOperator(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_stage(config: dict[str, Any], stage: str) -> Path:
    if stage not in {"sim", "real"}:
        raise ValueError("stage must be sim or real")
    seed_everything(int(config.get("seed", 20260808)))
    device = choose_device(config.get("device", "auto"))
    source = "sim" if stage == "sim" else "real"
    train_dataset = make_dataset(config, source, "train")
    validation_dataset = make_dataset(config, "real", "validation")
    train_loader = make_loader(train_dataset, config, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)

    model_config = ModelConfig.from_dict(config.get("model", {}), config.get("data", {}))
    model = HybridFlowOperator(model_config).to(device)
    if stage == "real":
        pretrained = torch.load(checkpoint_path(config, "sim"), map_location=device)
        model.load_state_dict(pretrained["state_dict"], strict=True)

    train_config = config["train"]
    steps = int(train_config[f"{stage}_steps"])
    learning_rate = float(
        train_config.get("finetune_learning_rate", 8e-5)
        if stage == "real"
        else train_config.get("learning_rate", 3e-4)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_config.get("amp", True)) and device.type == "cuda"
    )
    objective = MetricAwareLoss(LossWeights.from_dict(config.get("loss", {}))).to(device)
    ema = ExponentialMovingAverage(model, float(train_config.get("ema_decay", 0.9995)))
    batches = _forever(train_loader)
    best = float("inf")
    output = checkpoint_path(config, stage)
    log_path = Path(train_config.get("checkpoint_dir", "runs")) / f"{stage}_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    train_interval = bool(train_config.get("train_interval", True))
    throttle_seconds = max(0.0, float(train_config.get("throttle_seconds", 0.0)))
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            if train_interval:
                prediction, half_width = model(inputs, return_interval=True)
            else:
                prediction = model(inputs)
                half_width = None
            loss, components = objective(prediction, targets, half_width)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        if throttle_seconds:
            time.sleep(throttle_seconds)

        if step % int(train_config.get("log_every", 50)) == 0:
            record = {
                "stage": stage,
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))

        if step % int(train_config.get("val_every", 500)) == 0 or step == steps:
            metrics = validate(
                ema.shadow,
                validation_loader,
                device,
                max_batches=int(train_config.get("val_batches", 64)),
            )
            selection = metrics["relative_l2"] + 0.3 * metrics["tke"] + 0.2 * metrics["profile"]
            print(json.dumps({"stage": stage, "step": step, "validation": metrics}))
            if selection < best:
                best = selection
                save_checkpoint(output, ema.shadow, config, stage=stage, step=step, metrics=metrics)
            model.train()
    return output


@torch.inference_mode()
def validate_intervals(
    model: HybridFlowOperator,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate official SPS, weighted by the exact number of scored elements."""
    model.eval()
    model.interval_temperature.fill_(float(temperature))
    weighted_sps = 0.0
    weighted_coverage = 0.0
    scored_total = 0
    count = 0
    for count, (inputs, targets) in enumerate(loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
        scored_total += scored
        if max_batches is not None and count >= max_batches:
            break
    if count == 0 or scored_total == 0:
        raise ValueError("interval validation loader is empty")
    return {
        "sps_score": weighted_sps / scored_total,
        "coverage": weighted_coverage / scored_total,
    }


def train_uncertainty(config: dict[str, Any]) -> Path:
    """Freeze the point model and train only its conditional symmetric width head."""
    seed_everything(int(config.get("seed", 20260808)))
    device = choose_device(config.get("device", "auto"))
    train_dataset = make_dataset(config, "real", "train")
    validation_dataset = make_dataset(config, "real", "validation")
    train_loader = make_loader(train_dataset, config, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)

    source = checkpoint_path(config, "real")
    model = load_checkpoint(source, device)
    model.requires_grad_(False)
    model.width_head.requires_grad_(True)
    model.interval_temperature.fill_(1.0)
    model.eval()
    model.width_head.train()

    train_config = config["train"]
    steps = int(train_config.get("uncertainty_steps", 2000))
    quantile = float(train_config.get("interval_quantile", 0.9))
    optimizer = torch.optim.AdamW(
        model.width_head.parameters(),
        lr=float(train_config.get("uncertainty_learning_rate", 3e-4)),
        weight_decay=float(train_config.get("uncertainty_weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_config.get("amp", True)) and device.type == "cuda"
    )
    batches = _forever(train_loader)
    throttle_seconds = max(0.0, float(train_config.get("throttle_seconds", 0.0)))
    val_every = int(train_config.get("uncertainty_val_every", 250))
    max_val_batches = int(train_config.get("val_batches", 64))
    output = Path(train_config.get("uncertainty_checkpoint", source))
    log_path = (
        Path(train_config.get("checkpoint_dir", "runs")) / "uncertainty_history.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("-inf")

    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            prediction, half_width = model(inputs, return_interval=True)
            loss = interval_quantile_loss(
                prediction, targets, half_width, quantile=quantile
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if throttle_seconds:
            time.sleep(throttle_seconds)

        if step % int(train_config.get("log_every", 50)) == 0:
            record = {
                "stage": "uncertainty",
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                "interval_loss": float(loss.detach()),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))

        if step % val_every == 0 or step == steps:
            metrics = validate_intervals(
                model,
                validation_loader,
                device,
                temperature=1.0,
                max_batches=max_val_batches,
            )
            print(json.dumps({"stage": "uncertainty", "step": step, "validation": metrics}))
            if metrics["sps_score"] > best:
                best = metrics["sps_score"]
                save_checkpoint(
                    output,
                    model,
                    config,
                    stage="uncertainty",
                    step=step,
                    metrics=metrics,
                )
            model.eval()
            model.width_head.train()
    return output


@torch.inference_mode()
def calibrate_checkpoint(config: dict[str, Any]) -> Path:
    device = choose_device(config.get("device", "auto"))
    train_config = config["train"]
    source = Path(
        train_config.get("uncertainty_checkpoint", checkpoint_path(config, "real"))
    )
    if not source.exists():
        source = checkpoint_path(config, "real")
    model = load_checkpoint(source, device).eval()
    validation = make_dataset(config, "real", "validation")
    loader = make_loader(validation, config, shuffle=False)

    minimum = float(train_config.get("calibration_temperature_min", 0.25))
    maximum = float(train_config.get("calibration_temperature_max", 8.0))
    steps = int(train_config.get("calibration_temperature_steps", 25))
    candidates = torch.logspace(
        torch.log10(torch.tensor(minimum)),
        torch.log10(torch.tensor(maximum)),
        steps=steps,
    ).tolist()
    candidates.append(1.0)
    best_temperature = 1.0
    best_metrics = {"sps_score": float("-inf"), "coverage": 0.0}
    for temperature in sorted(set(float(value) for value in candidates)):
        metrics = validate_intervals(
            model,
            loader,
            device,
            temperature=temperature,
            max_batches=int(train_config.get("val_batches", 64)),
        )
        if metrics["sps_score"] > best_metrics["sps_score"]:
            best_temperature = temperature
            best_metrics = metrics

    model.interval_temperature.fill_(best_temperature)
    target = Path(config["submission"]["checkpoint"])
    calibration_metrics = {
        "interval_temperature": best_temperature,
        **best_metrics,
        "source_checkpoint": str(source),
    }
    save_checkpoint(
        target,
        model,
        config,
        stage="calibrated",
        step=0,
        metrics=calibration_metrics,
    )
    print(json.dumps({"calibrated": str(target), **calibration_metrics}, sort_keys=True))
    return target
