#!/usr/bin/env python3
"""Rebuild the fixed C180 method from released Track 1 data only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from realpde.config import load_config  # noqa: E402
from realpde.data import (  # noqa: E402
    CaseInfo,
    FlowWindowDataset,
    _field_group,
    discover_cases,
)
from realpde.fast_interval import FastIntervalConfig, FastIntervalHead  # noqa: E402
from realpde.flow_diagnostics import flow_diagnostics  # noqa: E402
from realpde.losses import LossWeights, MetricAwareLoss  # noqa: E402
from realpde.metrics import (  # noqa: E402
    SIGMA_GLOBAL,
    _mvpe_relative_l2_per_sample,
    _relative_l2_per_sample,
    _tke_relative_l2_per_sample,
    profile_relative_l2,
    relative_l2,
    tke_relative_l2,
)
from realpde.model import HybridFlowOperator, ModelConfig  # noqa: E402
from realpde.train import ExponentialMovingAverage, choose_device, seed_everything  # noqa: E402

CONFIG = ROOT / "configs/track1_final_retrain_c180.yaml"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        if tensor.is_complex():
            tensor = torch.view_as_real(tensor)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def all_finite(module: nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in module.state_dict().values())


def forever(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(workers) > 0,
        drop_last=bool(shuffle) and len(dataset) >= int(batch_size),
    )


def reynolds_folds(cases: Sequence[CaseInfo], folds: int) -> list[list[float]]:
    values = sorted({float(case.reynolds) for case in cases})
    quotient, remainder = divmod(len(values), int(folds))
    output: list[list[float]] = []
    offset = 0
    for fold in range(int(folds)):
        count = quotient + int(fold < remainder)
        output.append(values[offset : offset + count])
        offset += count
    if sorted(value for group in output for value in group) != values:
        raise RuntimeError("Reynolds folds do not partition the released real grid")
    return output


def flow_dataset(
    cases: Sequence[CaseInfo],
    config: dict[str, Any],
    *,
    split: str = "all",
    holdouts: Sequence[Sequence[float]] = (),
    stride: int,
) -> FlowWindowDataset:
    data = config["data"]
    return FlowWindowDataset(
        cases,
        in_steps=int(data["in_steps"]),
        out_steps=int(data["out_steps"]),
        stride=int(stride),
        spatial_stride=int(data["spatial_stride"]),
        split=split,
        holdout_regimes=holdouts,
    )


class Native79Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Use native 20->20 windows while preserving C180's 79-frame support."""

    def __init__(self, cases: Sequence[CaseInfo]) -> None:
        self.cases = list(cases)
        self.windows: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            self.windows.extend(
                (case_index, start)
                for start in range(0, case.frames - 79 + 1, 20)
            )
        if not self.windows:
            raise RuntimeError("no released real 79-frame windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_index, start = self.windows[index]
        case = self.cases[case_index]
        step = 2 if case.height > 32 or case.width > 64 else 1
        with h5py.File(case.path, "r") as handle:
            group = _field_group(handle)
            u = np.asarray(group["u"][start : start + 79, ::step, ::step], dtype=np.float32)
            v = np.asarray(group["v"][start : start + 79, ::step, ::step], dtype=np.float32)
            if case.has_pressure:
                p = np.asarray(group["p"][start : start + 79, ::step, ::step], dtype=np.float32)
            else:
                p = np.zeros_like(u)
        flow = torch.from_numpy(np.stack((u, v, p), axis=-1))
        if tuple(flow.shape) != (79, 32, 64, 3):
            raise RuntimeError(f"unexpected released flow shape: {tuple(flow.shape)}")
        native = flow[:40]
        return native[:20], native[20:], torch.tensor(index, dtype=torch.long)


def save_point(path: Path, model: HybridFlowOperator, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_point(path: Path, device: torch.device) -> HybridFlowOperator:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HybridFlowOperator(ModelConfig.from_dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def train_point(
    model: HybridFlowOperator,
    dataset: Dataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    minimum_learning_rate: float,
    seed: int,
    stage: str,
) -> tuple[HybridFlowOperator, dict[str, Any]]:
    seed_everything(seed)
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(steps), eta_min=float(minimum_learning_rate)
    )
    ema = ExponentialMovingAverage(model, float(config["training"]["ema_decay"]))
    objective = MetricAwareLoss(LossWeights.from_dict(config["loss"])).to(device)
    batches = forever(
        loader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + 101,
            workers=int(config["data"]["num_workers"]),
        )
    )
    started = time.perf_counter()
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        batch = next(batches)
        observed, target = batch[:2]
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        point, width = model(observed, return_interval=True)
        loss, components = objective(point, target, width)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        if step == 1 or step % 250 == 0 or step == int(steps):
            last = {
                "stage": stage,
                "step": step,
                "steps": int(steps),
                "learning_rate": scheduler.get_last_lr()[0],
                **components,
            }
            print(json.dumps(last, sort_keys=True), flush=True)
    trained = ema.shadow.eval().requires_grad_(False)
    return trained, {
        "stage": stage,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "training_windows": len(dataset),
        "runtime_seconds": time.perf_counter() - started,
        "final": last,
        "state_hash": state_hash(trained),
        "all_finite": all_finite(trained),
    }


@torch.inference_mode()
def cache_fold(
    model: HybridFlowOperator,
    validation: FlowWindowDataset,
    all_cases: Sequence[CaseInfo],
    device: torch.device,
    config: dict[str, Any],
    *,
    fold: int,
    held_reynolds: list[float],
    path: Path,
    batch_limit: int | None,
) -> tuple[Path, dict[str, Any]]:
    model.eval()
    global_regime = {case.regime: index for index, case in enumerate(all_cases)}
    names = (
        "diagnostics",
        "scale",
        "point",
        "target",
        "measured",
        "regime",
        "reynolds",
        "angle",
        "fold",
    )
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    data_loader = loader(
        validation,
        batch_size=int(config["training"]["oof_point"]["batch_size"]),
        shuffle=False,
        seed=int(config["seed"]) + 3000 + fold,
        workers=int(config["data"]["num_workers"]),
    )
    offset = 0
    for batch_index, (observed, target) in enumerate(data_loader, 1):
        count = len(observed)
        observed = observed.to(device, non_blocking=True)
        point = model(observed).float()
        diagnostics, scale = flow_diagnostics(observed, point)
        target_uv = target[..., :2].float()
        selected_cases = [
            validation.cases[validation.windows[index][0]]
            for index in range(offset, offset + count)
        ]
        parts["diagnostics"].append(diagnostics.half().cpu())
        parts["scale"].append(scale.float().cpu())
        parts["point"].append(point[..., :2].half().cpu())
        parts["target"].append(target_uv.half())
        parts["measured"].append(target_uv != 0.0)
        parts["regime"].append(
            torch.tensor([global_regime[case.regime] for case in selected_cases], dtype=torch.long)
        )
        parts["reynolds"].append(
            torch.tensor([case.reynolds for case in selected_cases], dtype=torch.float32)
        )
        parts["angle"].append(
            torch.tensor([case.angle for case in selected_cases], dtype=torch.float32)
        )
        parts["fold"].append(torch.full((count,), fold, dtype=torch.long))
        offset += count
        if batch_limit is not None and batch_index >= int(batch_limit):
            break
    payload = {name: torch.cat(values) for name, values in parts.items()}
    metadata = {
        "fold": fold,
        "held_reynolds": held_reynolds,
        "held_regimes": [list(case.regime) for case in validation.cases],
        "full_validation_windows": len(validation),
        "cached_windows": len(payload["fold"]),
        "cache_batch_limit": batch_limit,
    }
    payload["metadata"] = metadata
    torch.save(payload, path)
    return path, metadata


def load_oof(paths: Sequence[Path]) -> dict[str, torch.Tensor]:
    names = (
        "diagnostics",
        "scale",
        "point",
        "target",
        "measured",
        "regime",
        "reynolds",
        "angle",
        "fold",
    )
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for name in names:
            parts[name].append(payload[name])
    return {name: torch.cat(values) for name, values in parts.items()}


def sample_factor(point: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = _relative_l2_per_sample(point, target)
    tke = _tke_relative_l2_per_sample(point, target)
    mvpe = _mvpe_relative_l2_per_sample(point, target)
    return (
        0.5 * (1.0 - rel / (0.5 + rel))
        + 0.3 * (1.0 - tke / (0.5 + tke))
        + 0.2 * (1.0 - mvpe / (0.5 + mvpe))
    )


def point_objective(point: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        relative_l2(point, target)
        + 0.25 * tke_relative_l2(point, target)
        + 0.15 * profile_relative_l2(point, target)
    )


def initial_radius(cache: dict[str, torch.Tensor]) -> float:
    scale = cache["scale"]
    error = (cache["target"].float() - cache["point"].float()).abs() / scale
    values = error[cache["measured"]]
    if values.numel() > 4_000_000:
        values = values[:: math.ceil(values.numel() / 4_000_000)]
    return math.log(float(torch.quantile(values, 0.85).clamp(1e-3, 3.0)))


def train_head(
    cache: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
) -> tuple[FastIntervalHead, dict[str, Any]]:
    settings = config["training"]["interval_head"]
    seed = int(settings["seed"])
    seed_everything(seed)
    head_config = FastIntervalConfig(
        diagnostic_channels=int(cache["diagnostics"].shape[1]),
        hidden_channels=int(settings["hidden_channels"]),
        out_steps=int(cache["point"].shape[1]),
        maximum_normalized_shift=float(settings["maximum_normalized_shift"]),
        minimum_normalized_radius=float(settings["minimum_normalized_radius"]),
        maximum_normalized_radius=float(settings["maximum_normalized_radius"]),
        initial_log_radius=initial_radius(cache),
    )
    head = FastIntervalHead(head_config).to(device).train()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(steps), eta_min=float(settings["minimum_learning_rate"])
    )
    generator = torch.Generator().manual_seed(seed + 1)
    started = time.perf_counter()
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        selected = torch.randint(
            len(cache["point"]),
            (int(settings["batch_size"]),),
            generator=generator,
        )
        diagnostics = cache["diagnostics"].index_select(0, selected).to(device).float()
        scale = cache["scale"].index_select(0, selected).to(device)
        point = cache["point"].index_select(0, selected).to(device).float()
        target = cache["target"].index_select(0, selected).to(device).float()
        measured = cache["measured"].index_select(0, selected).to(device)
        optimizer.zero_grad(set_to_none=True)
        shift, radius = head(diagnostics)
        normalized_error = (target - point) / scale
        corrected = point + shift * scale
        location = F.smooth_l1_loss(
            shift[measured], normalized_error.clamp(-0.5, 0.5)[measured], beta=0.04
        )
        point_loss = point_objective(corrected, target)
        remaining = (normalized_error - shift).abs()
        delta = remaining - radius
        quantile = torch.maximum(0.88 * delta, -0.12 * delta)[measured].mean()
        soft_inside = torch.sigmoid((radius - remaining) / 0.04)
        factor = sample_factor(point, target).detach()
        reward = (
            factor[:, None, None, None, None]
            * soft_inside
            * torch.exp(-2.0 * radius * scale / SIGMA_GLOBAL)
        )[measured].mean()
        regularization = 0.002 * (
            shift.square().mean() + radius.clamp_min(1e-5).log().square().mean()
        )
        loss = (
            0.65 * location
            + 0.45 * point_loss
            + 0.12 * quantile
            - 0.75 * reward
            + regularization
        )
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), float(config["training"]["grad_clip"]))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 250 == 0 or step == int(steps):
            last = {
                "stage": "interval_head",
                "step": step,
                "steps": int(steps),
                "loss": float(loss.detach()),
                "soft_reward": float(reward.detach()),
            }
            print(json.dumps(last, sort_keys=True), flush=True)
    head.eval().requires_grad_(False)
    return head, {
        "stage": "interval_head",
        "steps": int(steps),
        "seed": seed,
        "oof_windows": len(cache["point"]),
        "runtime_seconds": time.perf_counter() - started,
        "model_config": head_config.to_dict(),
        "final": last,
        "state_hash": state_hash(head),
        "all_finite": all_finite(head),
    }


def joint_interval_objective(
    head: FastIntervalHead,
    observed: torch.Tensor,
    point: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diagnostics, scale = flow_diagnostics(observed, point)
    shift, radius = head(diagnostics)
    point_uv = point[..., :2]
    target_uv = target[..., :2]
    measured = target_uv != 0.0
    normalized_error = (target_uv - point_uv) / scale
    location = F.smooth_l1_loss(
        shift[measured], normalized_error.clamp(-0.5, 0.5)[measured], beta=0.04
    )
    remaining = (normalized_error - shift).abs()
    delta = remaining - radius
    quantile = torch.maximum(0.88 * delta, -0.12 * delta)[measured].mean()
    soft_inside = torch.sigmoid((radius - remaining) / 0.04)
    factor = sample_factor(point_uv, target_uv).detach()
    reward = (
        factor[:, None, None, None, None]
        * soft_inside
        * torch.exp(-2.0 * radius * scale / SIGMA_GLOBAL)
    )[measured].mean()
    regularization = 0.002 * (
        shift.square().mean() + radius.clamp_min(1e-5).log().square().mean()
    )
    loss = 0.65 * location + 0.12 * quantile - 0.75 * reward + regularization
    return loss, {"location": location, "quantile": quantile, "reward": reward}


def train_joint(
    source: Path,
    source_head: FastIntervalHead,
    cases: Sequence[CaseInfo],
    config: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
) -> tuple[HybridFlowOperator, FastIntervalHead, dict[str, Any], Native79Dataset]:
    settings = config["training"]["joint_fullfit"]
    seed = int(settings["seed"])
    seed_everything(seed)
    model = load_point(source, device).train().requires_grad_(True)
    head = copy.deepcopy(source_head).to(device).train().requires_grad_(True)
    dataset = Native79Dataset(cases)
    point_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["point_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    head_optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(settings["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    point_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        point_optimizer,
        T_max=int(steps),
        eta_min=float(settings["point_minimum_learning_rate"]),
    )
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        head_optimizer,
        T_max=int(steps),
        eta_min=float(settings["head_minimum_learning_rate"]),
    )
    point_ema = ExponentialMovingAverage(model, float(config["training"]["ema_decay"]))
    head_ema = ExponentialMovingAverage(head, float(config["training"]["ema_decay"]))
    objective = MetricAwareLoss(LossWeights.from_dict(config["loss"])).to(device)
    batches = forever(
        loader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            seed=seed + 202,
            workers=int(config["data"]["num_workers"]),
        )
    )
    started = time.perf_counter()
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        observed, target, _indices = next(batches)
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        point_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        point, width = model(observed, return_interval=True)
        point_loss, components = objective(point, target, width)
        interval_loss, interval_components = joint_interval_objective(
            head, observed, point, target
        )
        loss = point_loss + float(settings["interval_objective_weight"]) * interval_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        nn.utils.clip_grad_norm_(head.parameters(), float(config["training"]["grad_clip"]))
        point_optimizer.step()
        head_optimizer.step()
        point_scheduler.step()
        head_scheduler.step()
        point_ema.update(model)
        head_ema.update(head)
        if step == 1 or step % 250 == 0 or step == int(steps):
            last = {
                "stage": "joint_fullfit",
                "step": step,
                "steps": int(steps),
                "total": float(loss.detach()),
                "point": float(point_loss.detach()),
                "interval": float(interval_loss.detach()),
                "data": float(components["data"]),
                "tke": float(components["tke"]),
                "location": float(interval_components["location"].detach()),
                "soft_reward": float(interval_components["reward"].detach()),
            }
            print(json.dumps(last, sort_keys=True), flush=True)
    point_final = point_ema.shadow.eval().requires_grad_(False)
    head_final = head_ema.shadow.eval().requires_grad_(False)
    return point_final, head_final, {
        "stage": "joint_fullfit",
        "steps": int(steps),
        "seed": seed,
        "training_windows": len(dataset),
        "runtime_seconds": time.perf_counter() - started,
        "final": last,
        "point_state_hash": state_hash(point_final),
        "head_state_hash": state_hash(head_final),
        "point_all_finite": all_finite(point_final),
        "head_all_finite": all_finite(head_final),
    }, dataset


@torch.inference_mode()
def bounds_smoke(
    model: HybridFlowOperator,
    head: FastIntervalHead,
    dataset: Dataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    observed, target, _index = dataset[0]
    observed = observed.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    point = model(observed).float()[..., :2]
    diagnostics, scale = flow_diagnostics(observed, point)
    shift, radius = head(diagnostics)
    decision = config["fixed_decision"]
    group_widths = torch.tensor(
        decision["group_width_multipliers"], device=device, dtype=radius.dtype
    )[None, :, None, None, :]
    center = point + float(decision["center_multiplier"]) * shift * scale
    half_width = float(decision["width_multiplier"]) * radius * scale * group_widths
    lower = center - half_width
    upper = center + half_width
    return {
        "prediction_shape": list(point.shape),
        "target_shape": list(target[..., :2].shape),
        "all_finite": bool(
            torch.isfinite(point).all()
            and torch.isfinite(lower).all()
            and torch.isfinite(upper).all()
        ),
        "bounds_ordered": bool((lower <= upper).all()),
        "minimum_width": float((upper - lower).min()),
        "maximum_width": float((upper - lower).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--smoke-cache-batches", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if bool(config["training"]["mixed_precision"]):
        raise RuntimeError("fixed recipe requires FP32")
    if not bool(config["training"]["fixed_endpoint_no_validation_selection"]):
        raise RuntimeError("fixed endpoint declaration is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(config.get("device", "auto"))
    seed_everything(int(config["seed"]))
    random.seed(int(config["seed"]))

    data = config["data"]
    sim_cases = discover_cases(data["sim_dir"])
    real_cases = discover_cases(data["real_dir"], exclude_files=data["exclude_files"])
    sim_names = {case.path.name for case in sim_cases}
    real_names = {case.path.name for case in real_cases}
    expected_sim = int(data["expected_sim_cases"])
    expected_real = int(data["expected_real_cases"])
    if len(sim_cases) != expected_sim or len(real_cases) != expected_real:
        raise RuntimeError(
            f"released inventory mismatch: sim={len(sim_cases)}, real={len(real_cases)}"
        )
    if "7575_0.h5" in real_names or "6300_0.h5" not in real_names:
        raise RuntimeError("official exclusion/retention lineage mismatch")

    smoke_steps = int(args.smoke_steps)
    sim_steps = smoke_steps if args.smoke else int(config["training"]["sim"]["steps"])
    point_steps = smoke_steps if args.smoke else int(config["training"]["oof_point"]["steps"])
    head_steps = smoke_steps if args.smoke else int(config["training"]["interval_head"]["steps"])
    joint_steps = smoke_steps if args.smoke else int(config["training"]["joint_fullfit"]["steps"])
    cache_limit = int(args.smoke_cache_batches) if args.smoke else None

    sim_dataset = flow_dataset(
        sim_cases,
        config,
        stride=int(data["train_stride"]),
    )
    model_config = ModelConfig.from_dict(config["model"], config["data"])
    seed_everything(int(config["training"]["sim"]["seed"]))
    sim_model = HybridFlowOperator(model_config).to(device)
    sim_model, sim_report = train_point(
        sim_model,
        sim_dataset,
        config,
        device,
        steps=sim_steps,
        batch_size=int(config["training"]["sim"]["batch_size"]),
        learning_rate=float(config["training"]["sim"]["learning_rate"]),
        minimum_learning_rate=float(config["training"]["sim"]["minimum_learning_rate"]),
        seed=int(config["training"]["sim"]["seed"]),
        stage="sim_fixed_endpoint",
    )
    sim_path = args.output_dir / "sim_fixed_endpoint.pt"
    save_point(sim_path, sim_model, sim_report)
    del sim_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fold_groups = reynolds_folds(real_cases, int(config["training"]["oof_point"]["folds"]))
    cache_paths: list[Path] = []
    fold_reports: list[dict[str, Any]] = []
    covered_regimes: set[tuple[float, float]] = set()
    for fold, held_reynolds in enumerate(fold_groups):
        held = [
            [float(case.reynolds), float(case.angle)]
            for case in real_cases
            if float(case.reynolds) in set(held_reynolds)
        ]
        train = flow_dataset(
            real_cases,
            config,
            split="train",
            holdouts=held,
            stride=int(data["train_stride"]),
        )
        validation = flow_dataset(
            real_cases,
            config,
            split="validation",
            holdouts=held,
            stride=int(data["oof_validation_stride"]),
        )
        if set(case.regime for case in train.cases) & set(case.regime for case in validation.cases):
            raise RuntimeError(f"fold {fold} leaks a natural case")
        covered_regimes.update(case.regime for case in validation.cases)
        fold_seed = int(config["seed"]) + int(config["training"]["oof_point"]["seed_offset"]) + fold
        model = load_point(sim_path, device)
        model, point_report = train_point(
            model,
            train,
            config,
            device,
            steps=point_steps,
            batch_size=int(config["training"]["oof_point"]["batch_size"]),
            learning_rate=float(config["training"]["oof_point"]["learning_rate"]),
            minimum_learning_rate=float(config["training"]["oof_point"]["minimum_learning_rate"]),
            seed=fold_seed,
            stage=f"oof_point_fold_{fold}",
        )
        point_path = args.output_dir / f"fold_{fold}_point.pt"
        save_point(point_path, model, point_report)
        cache_path, cache_metadata = cache_fold(
            model,
            validation,
            real_cases,
            device,
            config,
            fold=fold,
            held_reynolds=held_reynolds,
            path=args.output_dir / f"fold_{fold}_cache.pt",
            batch_limit=cache_limit,
        )
        cache_paths.append(cache_path)
        fold_reports.append(
            {
                "fold": fold,
                "held_reynolds": held_reynolds,
                "training_cases": len(train.cases),
                "validation_cases": len(validation.cases),
                "natural_units_disjoint": True,
                "point": point_report,
                "cache": cache_metadata,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if covered_regimes != {case.regime for case in real_cases}:
        raise RuntimeError("five OOF folds do not cover clean81 exactly")

    oof = load_oof(cache_paths)
    head, head_report = train_head(oof, config, device, steps=head_steps)
    head_path = args.output_dir / "interval_head_pretrain.pt"
    torch.save(
        {
            "model_config": head.config.to_dict(),
            "state_dict": head.state_dict(),
            "metadata": head_report,
        },
        head_path,
    )

    final_model, final_head, joint_report, joint_dataset = train_joint(
        sim_path,
        head,
        real_cases,
        config,
        device,
        steps=joint_steps,
    )
    final_point_path = args.output_dir / "final_point.pt"
    final_head_path = args.output_dir / "final_head.pt"
    save_point(final_point_path, final_model, joint_report)
    torch.save(
        {
            "model_config": final_head.config.to_dict(),
            "state_dict": final_head.state_dict(),
            "metadata": joint_report,
        },
        final_head_path,
    )
    bounds = bounds_smoke(final_model, final_head, joint_dataset, config, device)

    decision = config["fixed_decision"]
    provenance = config["provenance"]
    validity = {
        "sim_cases_exact": len(sim_cases) == expected_sim,
        "real_cases_exact": len(real_cases) == expected_real,
        "excluded_file_absent": "7575_0.h5" not in real_names,
        "retained_file_present": "6300_0.h5" in real_names,
        "five_reynolds_folds": len(fold_reports) == 5,
        "oof_covers_all_clean_regimes": len(covered_regimes) == expected_real,
        "all_fold_natural_units_disjoint": all(row["natural_units_disjoint"] for row in fold_reports),
        "all_four_stages_exercised": bool(
            sim_report["steps"] > 0
            and all(row["point"]["steps"] > 0 for row in fold_reports)
            and head_report["steps"] > 0
            and joint_report["steps"] > 0
        ),
        "all_states_finite": bool(
            sim_report["all_finite"]
            and all(row["point"]["all_finite"] for row in fold_reports)
            and head_report["all_finite"]
            and joint_report["point_all_finite"]
            and joint_report["head_all_finite"]
        ),
        "bounds_finite_and_ordered": bounds["all_finite"] and bounds["bounds_ordered"],
        "decision_shape_exact": len(decision["group_width_multipliers"]) == 20
        and all(len(row) == 2 for row in decision["group_width_multipliers"]),
        "scratch_inputs_release_only": provenance["scratch_training_data"]
        == "released_sim100_and_clean81_real_only",
        "random_initial_weights": provenance["scratch_initial_weights"]
        == "random_initialization_only",
        "no_prior_artifact_inputs": not provenance["prior_submission_or_run_artifacts_as_inputs"],
        "truthful_historical_selection_disclosure": bool(
            provenance["historical_development_public_used_to_freeze_interval_constants"]
        )
        and not provenance["constants_reselected_during_organizer_retraining"],
    }
    status = "INVALID"
    if all(validity.values()):
        status = "SMOKE_PASS" if args.smoke else "FULL_COMPLETE"
    report = {
        "card_id": "C252",
        "status": status,
        "mode": "smoke" if args.smoke else "full",
        "config_sha256": file_hash(args.config),
        "device": str(device),
        "inventory": {
            "sim_cases": len(sim_cases),
            "real_cases": len(real_cases),
            "sim_names_sha256": hashlib.sha256("\n".join(sorted(sim_names)).encode()).hexdigest(),
            "real_names_sha256": hashlib.sha256("\n".join(sorted(real_names)).encode()).hexdigest(),
            "excluded": data["exclude_files"],
            "retained": data["retained_duplicate_source"],
        },
        "steps": {
            "sim": sim_steps,
            "oof_point_per_fold": point_steps,
            "interval_head": head_steps,
            "joint_fullfit": joint_steps,
        },
        "sim": sim_report,
        "folds": fold_reports,
        "interval_head": head_report,
        "joint_fullfit": joint_report,
        "bounds": bounds,
        "fixed_decision": decision,
        "provenance": provenance,
        "learned_state_hashes": {
            "sim": sim_report["state_hash"],
            "oof_points": [row["point"]["state_hash"] for row in fold_reports],
            "interval_head": head_report["state_hash"],
            "final_point": joint_report["point_state_hash"],
            "final_head": joint_report["head_state_hash"],
        },
        "validity": validity,
        "validity_pass": all(validity.values()),
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "validity_pass": report["validity_pass"],
        "learned_state_hashes": report["learned_state_hashes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
