"""Portable identity interval learning with a separate frozen center head.

No experiment, saved historical run, router or fitted deployment-map dependency.
The caller owns source/data authorization and enforces runtime/disk/RAM budgets.
Importing this module does not change process-wide determinism or CUDA settings.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import CaseInfo, _field_group
from .fast_interval import FastIntervalConfig, FastIntervalHead
from .flow_diagnostics import flow_diagnostics
from .metrics import (SIGMA_GLOBAL, _relative_l2_per_sample,
                      _tke_relative_l2_per_sample, _mvpe_relative_l2_per_sample)
from .model import HybridFlowOperator, ModelConfig
from .train import ExponentialMovingAverage

ARMS = ("original", "raw", "free", "preserved")
SEED = 2026090541
STEPS = 2000
BATCH = 8


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    if value.is_complex():
        value = torch.view_as_real(value)
    digest = hashlib.sha256()
    data = memoryview(value.numpy()).cast("B")
    for start in range(0, len(data), 1 << 20):
        digest.update(data[start:start + (1 << 20)])
    return digest.hexdigest()


def state_hash(model):
    """The original scratch recipe's raw-byte state hash, without large copies."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        value = value.detach().cpu().contiguous()
        if value.is_complex():
            value = torch.view_as_real(value)
        data = memoryview(value.numpy()).cast("B")
        for start in range(0, len(data), 1 << 20):
            digest.update(data[start:start + (1 << 20)])
    return digest.hexdigest()


def all_finite(model):
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def log(payload):
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def load_point(path, device):
    value = torch.load(path, map_location="cpu", weights_only=False)
    model = HybridFlowOperator(ModelConfig.from_dict(value["model_config"])).to(device)
    model.load_state_dict(value["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def load_head(path, device):
    value = torch.load(path, map_location="cpu", weights_only=False)
    head = FastIntervalHead(FastIntervalConfig.from_dict(value["model_config"])).to(device)
    head.load_state_dict(value["state_dict"], strict=True)
    return head.eval().requires_grad_(False)


def temperature(step):
    if not 1 <= step <= 2000:
        raise ValueError("step must belong to the fixed 1..2000 schedule")
    return .04 * 10.0 ** (-(step - 1) / 1999)


def data_settings(config):
    """Only dataset configuration; fitted deployment settings are not consumed."""
    return copy.deepcopy(config["data"])


def sample_factor(point, target):
    rel = _relative_l2_per_sample(point, target)
    tke = _tke_relative_l2_per_sample(point, target)
    mvpe = _mvpe_relative_l2_per_sample(point, target)
    return (
        0.5 * (1.0 - rel / (0.5 + rel))
        + 0.3 * (1.0 - tke / (0.5 + tke))
        + 0.2 * (1.0 - mvpe / (0.5 + mvpe))
    )


def identity_objective(head, batch, tau, frozen_shift=None):
    candidate_shift, radius = head(batch["diagnostics"])
    shift = candidate_shift if frozen_shift is None else frozen_shift.detach()
    measured = batch["target"] != 0.0
    error = (batch["target"] - batch["point"]) / batch["scale"]
    location = F.smooth_l1_loss(shift[measured], error.clamp(-.5, .5)[measured], beta=.04)
    remaining = (error - shift).abs()
    delta = remaining - radius
    quantile = torch.maximum(.88 * delta, -.12 * delta)[measured].mean()
    factor = sample_factor(batch["point"], batch["target"]).detach()
    reward = (factor[:, None, None, None, None] * torch.sigmoid((radius - remaining) / tau)
              * torch.exp(-2.0 * radius * batch["scale"] / SIGMA_GLOBAL))[measured].mean()
    regularization = .002 * (shift.square().mean() + radius.clamp_min(1e-5).log().square().mean())
    loss = .65 * location + .12 * quantile - .75 * reward + regularization
    return loss, {"loss": loss, "location": location, "quantile": quantile,
                  "reward": reward, "regularization": regularization}


def identity_bounds(point, scale, shift, radius):
    center = point + shift * scale
    half_width = radius * scale
    return center, center - half_width, center + half_width


def gradient_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(b"NONE" if parameter.grad is None else parameter.grad.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(model):
    terms = [p.grad.detach().double().square().sum() for p in model.parameters() if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def float_components(values):
    return {name: float(value.detach()) for name, value in values.items()}


class Native79Dataset(Dataset):
    """Exact native 20->20 windows within the scratch recipe's 79-frame support."""

    def __init__(self, cases: Sequence[CaseInfo]):
        self.cases = list(cases)
        self.windows = [(index, start) for index, case in enumerate(self.cases)
                        for start in range(0, case.frames - 79 + 1, 20)]
        if not self.windows:
            raise RuntimeError("no released real 79-frame windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        case_index, start = self.windows[index]
        case = self.cases[case_index]
        step = 2 if case.height > 32 or case.width > 64 else 1
        with h5py.File(case.path, "r") as handle:
            group = _field_group(handle)
            u = np.asarray(group["u"][start:start + 79, ::step, ::step], dtype=np.float32)
            v = np.asarray(group["v"][start:start + 79, ::step, ::step], dtype=np.float32)
            if case.has_pressure:
                p = np.asarray(group["p"][start:start + 79, ::step, ::step], dtype=np.float32)
            else:
                p = np.zeros_like(u)
        flow = torch.from_numpy(np.stack((u, v, p), axis=-1))
        if tuple(flow.shape) != (79, 32, 64, 3):
            raise RuntimeError(f"unexpected released flow shape: {tuple(flow.shape)}")
        native = flow[:40]
        return native[:20], native[20:], torch.tensor(index, dtype=torch.long)


def training_cache(model, dataset, device, count, budget, *, max_cache_bytes=8 * 1024**3):
    """Fresh FP32 batch-eight cache; no precision compression or disk cache."""
    if not 1 <= count <= len(dataset):
        raise ValueError("requested training-cache count outside dataset")
    cache = {}
    started = time.monotonic()
    for start in range(0, count, BATCH):
        budget.check()
        stop = min(count, start + BATCH)
        items = [dataset[index] for index in range(start, stop)]
        observed = torch.stack([item[0] for item in items]).float()
        target = torch.stack([item[1][..., :2] for item in items]).float()
        with budget.cuda(), torch.inference_mode():
            obs_gpu = observed.to(device)
            point = model(obs_gpu).float()[..., :2]
            diagnostics, scale = flow_diagnostics(obs_gpu, point)
        values = {"point": point.cpu(), "target": target,
                  "diagnostics": diagnostics.cpu(), "scale": scale.cpu()}
        if start == 0:
            required = sum(value[0].numel() * 4 * count for value in values.values())
            if required > max_cache_bytes:
                raise RuntimeError("training cache exceeds declared RAM cap")
        for name, value in values.items():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"invalid float32 training cache: {name}")
            if name not in cache:
                # Allocate outside inference mode: convolution backward saves it.
                cache[name] = torch.empty((count, *value.shape[1:]), dtype=torch.float32)
            cache[name][start:stop].copy_(value)
        if start == 0 or stop == count or stop % 256 == 0:
            log({"stage": "train_cache", "rows": stop, "total": count})
    return cache, {"rows": count, "dtype": "float32", "persistent": False,
                   "bytes": sum(value.numel() * value.element_size() for value in cache.values()),
                   "seconds": time.monotonic() - started,
                   "target_sha256": tensor_hash(cache["target"]), "point_sha256": tensor_hash(cache["point"]),
                   "scale_sha256": tensor_hash(cache["scale"]),
                   "diagnostics_sha256": tensor_hash(cache["diagnostics"])}


def train_three(original, cache, device, budget, steps, output, *, source_head_sha256, card_id="C543"):
    models = {arm: copy.deepcopy(original).train().requires_grad_(True) for arm in ARMS[1:]}
    initial_hash = state_hash(original)
    if any(state_hash(model) != initial_hash for model in models.values()):
        raise RuntimeError("three initial heads not identical")
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4) for arm, model in models.items()}
    schedules = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5) for arm, opt in optimizers.items()}
    emas = {arm: ExponentialMovingAverage(model, .9995) for arm, model in models.items()}
    rng = torch.Generator().manual_seed(SEED)
    batch_digests = {arm: hashlib.sha256() for arm in models}
    gradient_max = {arm: 0.0 for arm in models}
    history, first = [], {}
    began = time.monotonic()
    with (output / "training_curve.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(1, steps + 1):
            budget.check()
            indices = torch.randint(len(cache["point"]), (8,), generator=rng)
            batch = {name: value.index_select(0, indices).to(device) for name, value in cache.items()}
            with budget.cuda(), torch.no_grad():
                frozen_shift, _ = original(batch["diagnostics"])
            row = {"step": step, "steps": steps, "scheduled_tau": temperature(step)}
            first_gradients = {}
            for arm, head in models.items():
                batch_digests[arm].update(indices.numpy().tobytes())
                tau = .04 if arm == "raw" else temperature(step)
                with budget.cuda():
                    optimizers[arm].zero_grad(set_to_none=True)
                    loss, values = identity_objective(head, batch, tau, frozen_shift if arm == "preserved" else None)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"nonfinite {arm} loss at step {step}")
                    loss.backward()
                    norm = gradient_norm(head)
                    if not np.isfinite(norm) or norm <= 0:
                        raise RuntimeError(f"inactive/nonfinite {arm} gradient at step {step}")
                    gradient_max[arm] = max(gradient_max[arm], norm)
                    if step == 1:
                        first_gradients[arm] = gradient_hash(head)
                    nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    optimizers[arm].step()
                    schedules[arm].step()
                    emas[arm].update(head)
                row[arm] = {**float_components(values), "gradient_norm_before_clip": norm,
                            "learning_rate_after_update": schedules[arm].get_last_lr()[0]}
            if step == 1:
                keys = ("loss", "location", "quantile", "reward", "regularization")
                first = {"raw_free_loss_components_exact": all(row["raw"][k] == row["free"][k] for k in keys),
                         "free_preserved_initial_loss_components_exact": all(row["free"][k] == row["preserved"][k] for k in keys),
                         "raw_free_all_gradient_tensors_exact": first_gradients["raw"] == first_gradients["free"],
                         "preserved_gradient_path_differs": first_gradients["preserved"] != first_gradients["free"],
                         "raw_free_first_optimizer_update_exact": state_hash(models["raw"]) == state_hash(models["free"]),
                         "raw_free_first_ema_update_exact": state_hash(emas["raw"].shadow) == state_hash(emas["free"].shadow),
                         "gradient_sha256": first_gradients}
                if not all(value for key, value in first.items() if key != "gradient_sha256"):
                    raise RuntimeError({"step1_method_parity_failed": first})
            if step <= 2 or step % 100 == 0 or step == steps:
                for arm, ema in emas.items():
                    with budget.cuda(), torch.no_grad():
                        _, values = identity_objective(ema.shadow, batch, .04 if arm == "raw" else temperature(step), frozen_shift if arm == "preserved" else None)
                        _, fixed_values = identity_objective(ema.shadow, batch, .04, frozen_shift if arm == "preserved" else None)
                    row[arm]["ema_same_batch_after_update"] = float_components(values)
                    row[arm]["ema_tau_0_04_diagnostic"] = float_components(fixed_values)
                history.append(row)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                log({"card_id": card_id, "stage": "paired_train", **row})
                budget.check(disk=True)
    endpoints = {arm: ema.shadow.eval().requires_grad_(False) for arm, ema in emas.items()}
    for arm, head in endpoints.items():
        if not all_finite(head):
            raise RuntimeError("nonfinite final EMA")
        torch.save({"card_id": card_id, "arm": arm, "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "model_config": head.config.to_dict(), "source_head_sha256": source_head_sha256,
                    "steps": steps, "seed": SEED, "selection": "fixed_final_EMA_only",
                    "adopted_center_source": "frozen_original_head" if arm == "preserved" else "candidate_head",
                    "identity_map_only": True}, output / f"{arm}_ema_head.pt")
    return endpoints, {"steps_per_arm": steps, "seed": SEED, "batch_size": 8, "learning_rate": 4e-4,
                       "minimum_learning_rate": 1e-5, "cosine_T_max": 2000, "weight_decay": 1e-4, "gradient_clip": 1.,
                       "ema_decay": .9995, "first_step": first, "initial_state_sha256": initial_hash,
                       "endpoint_state_sha256": {a: state_hash(m) for a, m in endpoints.items()},
                       "batch_sha256": {a: h.hexdigest() for a, h in batch_digests.items()}, "gradient_max": gradient_max,
                       "history": history, "seconds": time.monotonic() - began, "held_targets_read_during_training": False}



