from __future__ import annotations

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import profile_relative_l2, relative_l2, tke_relative_l2
from .model import TKEMapConfig, TKEMapCorrector
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


def save_tke_map_checkpoint(
    path: str | Path,
    model: TKEMapCorrector,
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
            "format_version": 5,
            "model_type": "tke_map_corrector",
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


def load_tke_map_checkpoint(path: str | Path, device: torch.device) -> TKEMapCorrector:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "tke_map_corrector":
        raise ValueError(f"not a tke_map_corrector checkpoint: {path}")
    model = TKEMapCorrector(TKEMapConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def initialize_tke_map(config: dict[str, Any], device: torch.device) -> TKEMapCorrector:
    values = dict(config["tke_corrector"])
    base_checkpoint = torch.load(values["base_checkpoint"], map_location=device)
    if base_checkpoint.get("model_type") != "masked_spectral_posterior":
        raise ValueError("TKE base must be a masked_spectral_posterior checkpoint")
    values["base"] = base_checkpoint["model_config"]
    model = TKEMapCorrector(TKEMapConfig.from_dict(values)).to(device)
    model.base.load_state_dict(base_checkpoint["state_dict"], strict=True)
    model.base.requires_grad_(False)
    model.base.eval()
    return model


@torch.inference_mode()
def _cache_tke_components(
    model: TKEMapCorrector, dataset, device: torch.device, batch_size: int
) -> dict[str, Any]:
    model.eval()
    cached: dict[str, list[torch.Tensor]] = {
        "base": [],
        "width": [],
        "log_ratio": [],
        "target": [],
    }
    regimes: list[tuple[float, float]] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        samples = [dataset[index] for index in indices]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        targets = torch.stack([sample[1] for sample in samples]).to(device)
        base, width, log_ratio = model.components(inputs)
        cached["base"].append(base)
        cached["width"].append(width)
        cached["log_ratio"].append(log_ratio)
        cached["target"].append(targets)
        regimes.extend(dataset.cases[dataset.windows[index][0]].regime for index in indices)
    result: dict[str, Any] = {name: torch.cat(parts) for name, parts in cached.items()}
    result["regimes"] = regimes
    return result


@torch.inference_mode()
def evaluate_tke_candidates(
    model: TKEMapCorrector,
    dataset,
    device: torch.device,
    values: dict[str, Any],
) -> dict[str, Any]:
    cached = _cache_tke_components(
        model, dataset, device, int(values.get("validation_batch_size", 2))
    )
    scales = sorted(
        {0.0, *(float(value) for value in values.get("correction_scales", [0, 0.5, 1]))}
    )
    width_scales = sorted(
        {float(value) for value in values.get("interval_width_scales", [0.9, 1.0, 1.1])}
    )
    runtime_score = float(values.get("runtime_score_assumption", 90.0))
    runtime_family = str(values.get("runtime_family", "one_pass"))
    transfer_path = Path(values.get("public_transfer_config", "configs/public_transfer_v9.json"))
    transfer = PublicTransferModel.load(transfer_path) if transfer_path.exists() else None
    unique_regimes = sorted(set(cached["regimes"]))
    candidates: list[dict[str, Any]] = []
    for scale in scales:
        point = model.apply_log_tke_ratio(cached["base"], cached["log_ratio"], scale)
        for width_scale in width_scales:
            width = cached["width"] * width_scale
            metrics = _score_candidate(
                point[..., :2], width[..., :2], cached["target"], runtime_score
            )
            case_finals: dict[str, float] = {}
            case_tke: dict[str, float] = {}
            for regime in unique_regimes:
                indices = [
                    index for index, observed in enumerate(cached["regimes"]) if observed == regime
                ]
                selected = torch.tensor(indices, device=device)
                case = _score_candidate(
                    point.index_select(0, selected)[..., :2],
                    width.index_select(0, selected)[..., :2],
                    cached["target"].index_select(0, selected),
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
            if transfer is None:
                proxy = None
                proxy_final = metrics["inferred_final_score"]
            else:
                proxy = transfer.estimate(
                    LocalScores(
                        relative_l2=metrics["relative_l2"],
                        tke=metrics["tke"],
                        mvpe=metrics["profile"],
                        time_score=runtime_score,
                        sps_score=metrics["sps_score"],
                    ),
                    runtime_family=runtime_family,
                )
                proxy_final = float(proxy["scores"]["final_score"])
            candidates.append(
                {
                    "correction_scale": scale,
                    "interval_width_scale": width_scale,
                    **metrics,
                    "case_final": case_finals,
                    "case_tke": case_tke,
                    "case_median_final": statistics.median(finals),
                    "case_worst_final": min(finals),
                    "robust_local_score": robust,
                    "public_proxy": proxy,
                    "public_proxy_final": proxy_final,
                    "selection": 0.80 * proxy_final + 0.20 * robust,
                }
            )
    baseline = max(
        (item for item in candidates if item["correction_scale"] == 0.0),
        key=lambda item: item["selection"],
    )
    tolerance = float(values.get("local_score_tolerance", 0.02))
    eligible = [
        item
        for item in candidates
        if item["inferred_final_score"] >= baseline["inferred_final_score"] - tolerance
    ]
    best = max(eligible, key=lambda item: item["selection"])
    for item in candidates:
        item["local_delta_from_base"] = (
            item["inferred_final_score"] - baseline["inferred_final_score"]
        )
        item["tke_delta_from_base"] = item["tke"] - baseline["tke"]
    ordered = sorted(candidates, key=lambda item: item["selection"], reverse=True)
    return {
        "sample_count": len(cached["regimes"]),
        "baseline": baseline,
        "best": best,
        "top_candidates": ordered[: int(values.get("report_top", 12))],
    }


def _tke_training_loss(
    model: TKEMapCorrector,
    base: torch.Tensor,
    log_ratio: torch.Tensor,
    target: torch.Tensor,
    values: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base_tke = model.temporal_tke(base).clamp_min(1e-7)
    true_tke = model.temporal_tke(target).clamp_min(1e-7)
    maximum = float(model.config.maximum_log_ratio)
    target_log_ratio = torch.log(true_tke / base_tke).clamp(-maximum, maximum)
    predicted = log_ratio[:, 0]
    importance = true_tke / true_tke.mean(dim=(1, 2), keepdim=True).clamp_min(1e-7)
    importance = importance.clamp(0.25, 4.0).sqrt()
    map_loss = (F.smooth_l1_loss(predicted, target_log_ratio, reduction="none") * importance).mean()
    point = model.apply_log_tke_ratio(base, log_ratio, 1.0)
    smooth_y = (predicted[:, 1:] - predicted[:, :-1]).abs().mean()
    smooth_x = (predicted[:, :, 1:] - predicted[:, :, :-1]).abs().mean()
    components = {
        "map": map_loss,
        "tke": tke_relative_l2(point, target),
        "data": relative_l2(point, target),
        "profile": profile_relative_l2(point, target),
        "smooth": smooth_y + smooth_x,
        "ratio": predicted.square().mean(),
    }
    weights = {
        "map": float(values.get("map_weight", 1.0)),
        "tke": float(values.get("tke_weight", 1.0)),
        "data": float(values.get("data_weight", 0.2)),
        "profile": float(values.get("profile_weight", 0.05)),
        "smooth": float(values.get("smooth_weight", 0.02)),
        "ratio": float(values.get("ratio_weight", 0.002)),
    }
    return sum(weights[name] * value for name, value in components.items()), components


def train_tke_map(config: dict[str, Any], stage: str) -> Path:
    if stage not in {"sim", "real"}:
        raise ValueError("TKE stage must be sim or real")
    seed_everything(int(config.get("seed", 20260810)) + (4 if stage == "sim" else 5))
    device = choose_device(config.get("device", "auto"))
    values = config["tke_corrector"]
    if stage == "sim":
        model = initialize_tke_map(config, device)
    else:
        source = Path(values["sim_checkpoint"])
        model = (
            load_tke_map_checkpoint(source, device)
            if source.exists()
            else initialize_tke_map(config, device)
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    train_loader = make_loader(
        make_dataset(config, "sim" if stage == "sim" else "real", "train"),
        config,
        shuffle=True,
        batch_size=int(values.get("batch_size", 2)),
    )
    validation_dataset = make_dataset(config, "real", "validation")
    batches = _forever(train_loader)
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
    amp_enabled = amp_requested and not has_complex_parameters
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if amp_requested and has_complex_parameters:
        print(
            json.dumps(
                {
                    "stage": f"tke_map_{stage}",
                    "amp_enabled": False,
                    "reason": "complex spectral parameters require FP32 gradient scaling",
                }
            ),
            flush=True,
        )
    ema = ExponentialMovingAverage(model, float(values.get("ema_decay", 0.995)))
    output = Path(values[f"{stage}_checkpoint"])
    candidate_dir = Path(values.get("candidates_dir", "checkpoints/tke_map_candidates"))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(values.get("history_dir", "runs/track1_tke_map"))
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{stage}_history.jsonl"
    best_selection = float("-inf")

    def validate_and_save(step: int) -> None:
        nonlocal best_selection
        report = evaluate_tke_candidates(ema.shadow, validation_dataset, device, values)
        chosen = report["best"]
        ema.shadow.config = replace(
            ema.shadow.config,
            correction_scale=float(chosen["correction_scale"]),
            interval_width_scale=float(chosen["interval_width_scale"]),
        )
        if stage == "real":
            save_tke_map_checkpoint(
                candidate_dir / f"step_{step:04d}.pt",
                ema.shadow,
                config,
                stage="tke_map_candidate",
                step=step,
                metrics=report,
            )
        if float(chosen["selection"]) > best_selection:
            best_selection = float(chosen["selection"])
            save_tke_map_checkpoint(
                output,
                ema.shadow,
                config,
                stage=f"tke_map_{stage}",
                step=step,
                metrics=report,
            )
        print(
            json.dumps({"stage": f"tke_map_{stage}", "step": step, "validation": report}),
            flush=True,
        )

    validate_and_save(0)
    model.train()
    log_every = int(values.get("log_every", 25))
    val_every = int(values.get("val_every", 250))
    throttle = max(0.0, float(values.get("throttle_seconds", 0.02)))
    for step in range(1, steps + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            base, _, log_ratio = model.components(inputs)
            loss, components = _tke_training_loss(model, base, log_ratio, targets, values)
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
                "stage": f"tke_map_{stage}",
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
def evaluate_tke_map(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config.get("device", "auto"))
    values = config["tke_corrector"]
    checkpoint = Path(config["submission"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = Path(values["real_checkpoint"])
    model = load_tke_map_checkpoint(checkpoint, device).eval()
    report = evaluate_tke_candidates(
        model, make_dataset(config, "real", "validation"), device, values
    )
    result = {"checkpoint": str(checkpoint), "validation": report}
    print(json.dumps(result, indent=2), flush=True)
    return result
