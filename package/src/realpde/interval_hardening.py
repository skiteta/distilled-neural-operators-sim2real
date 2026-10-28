"""Release-only fixed interval-head continuation; no historical experiment inputs."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import load_config
from .data import _field_group, discover_cases
from .fast_interval import FastIntervalConfig, FastIntervalHead
from .flow_diagnostics import flow_diagnostics
from .metrics import SIGMA_GLOBAL, _relative_l2_per_sample, _tke_relative_l2_per_sample, _mvpe_relative_l2_per_sample
from .model import HybridFlowOperator, ModelConfig
from .train import ExponentialMovingAverage, choose_device, seed_everything

SEED = 2026090504
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
    data = memoryview(value.numpy()).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(data), 1 << 20):
        digest.update(data[start:start + (1 << 20)])
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def temperature(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step must belong to the fixed 1..2000 schedule")
    return 0.04 * 10.0 ** (-(step - 1) / 1999.0)


def sample_factor(point, target):
    metrics = (_relative_l2_per_sample(point, target),
               _tke_relative_l2_per_sample(point, target),
               _mvpe_relative_l2_per_sample(point, target))
    return sum(weight * (1 - metric / (0.5 + metric))
               for weight, metric in zip((0.5, 0.3, 0.2), metrics)).detach()


def objective(head, batch, group, tau):
    raw_shift, raw_radius = head(batch["diagnostics"])
    regularization = 0.002 * (raw_shift.square().mean() + raw_radius.clamp_min(1e-5).log().square().mean())
    shift, radius = 1.5 * raw_shift, raw_radius * group
    normalized_error = (batch["target"] - batch["point"]) / batch["scale"]
    measured = batch["target"] != 0.0
    factor = sample_factor(batch["point"], batch["target"])
    location = F.smooth_l1_loss(shift[measured], normalized_error.clamp(-0.5, 0.5)[measured], beta=0.04)
    remaining = (normalized_error - shift).abs()
    delta = remaining - radius
    quantile = torch.maximum(0.88 * delta, -0.12 * delta)[measured].mean()
    reward = (factor[:, None, None, None, None]
              * torch.sigmoid((radius - remaining) / tau)
              * torch.exp(-2 * radius * batch["scale"] / SIGMA_GLOBAL))[measured].mean()
    loss = 0.65 * location + 0.12 * quantile - 0.75 * reward + regularization
    return loss, {"loss": float(loss.detach()), "location": float(location.detach()),
                  "quantile": float(quantile.detach()), "reward": float(reward.detach()),
                  "regularization": float(regularization.detach())}


class Native79Dataset(Dataset):
    """Exact released native 20->20 support used by the original scratch recipe."""
    def __init__(self, cases):
        self.cases = list(cases)
        self.windows = [(i, start) for i, case in enumerate(self.cases)
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
            p = (np.asarray(group["p"][start:start + 79, ::step, ::step], dtype=np.float32)
                 if case.has_pressure else np.zeros_like(u))
        flow = torch.from_numpy(np.stack((u, v, p), axis=-1))
        if tuple(flow.shape) != (79, 32, 64, 3):
            raise RuntimeError(f"unexpected released flow shape: {tuple(flow.shape)}")
        native = flow[:40]
        return native[:20], native[20:], torch.tensor(index, dtype=torch.long)


class Budget:
    def __init__(self, output, device):
        self.output, self.device = Path(output), device
        self.started = time.monotonic()
        self.gpu_seconds = 0.0

    def check(self):
        if time.monotonic() - self.started > 7200 or self.gpu_seconds > 3600:
            raise TimeoutError("fixed hardening runtime budget exceeded")
        if sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) > 3 * 1024 ** 3:
            raise RuntimeError("fixed hardening persisted-artifact budget exceeded")

    @contextmanager
    def cuda(self):
        self.check()
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                end.record()
                end.synchronize()
                self.gpu_seconds += start.elapsed_time(end) / 1000.0
            self.check()


def training_cache(point_model, dataset, device, count, budget):
    cache = {}
    started = time.monotonic()
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        items = [dataset[i] for i in range(start, stop)]
        observed = torch.stack([item[0] for item in items]).float()
        target = torch.stack([item[1][..., :2] for item in items]).float()
        with budget.cuda(), torch.inference_mode():
            observed_gpu = observed.to(device)
            point = point_model(observed_gpu).float()[..., :2]
            diagnostics, scale = flow_diagnostics(observed_gpu, point)
        values = {"point": point.cpu(), "target": target, "diagnostics": diagnostics.cpu(), "scale": scale.cpu()}
        for name, value in values.items():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"invalid FP32 cache field {name}")
            if name not in cache:
                cache[name] = torch.empty((count, *value.shape[1:]), dtype=torch.float32)
            cache[name][start:stop].copy_(value)
        if start == 0 or stop == count or stop % 256 == 0:
            print(json.dumps({"stage": "hardening_cache", "rows": stop, "total": count}), flush=True)
    size = sum(value.numel() * value.element_size() for value in cache.values())
    if size > 12 * 1024 ** 3:
        raise RuntimeError("hardening RAM cache exceeds 12 GiB")
    return cache, {"rows": count, "dtype": "float32", "bytes": size, "persistent": False,
                   "seconds": time.monotonic() - started,
                   "sha256": {name: tensor_hash(value) for name, value in cache.items()}}


def make_optimizer(head):
    optimizer = torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
    return optimizer, scheduler, ExponentialMovingAverage(head, 0.9995)


def train_endpoint(initial, cache, group, device, output, budget, steps):
    head = copy.deepcopy(initial).train().requires_grad_(True)
    optimizer, scheduler, ema = make_optimizer(head)
    generator = torch.Generator().manual_seed(SEED)
    batch_hash = hashlib.sha256()
    history = []
    max_gradient = 0.0
    started = time.monotonic()
    with (Path(output) / "training_curve.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(1, steps + 1):
            indices = torch.randint(len(cache["point"]), (BATCH,), generator=generator)
            batch_hash.update(indices.numpy().tobytes())
            batch = {name: value.index_select(0, indices).to(device) for name, value in cache.items()}
            with budget.cuda():
                optimizer.zero_grad(set_to_none=True)
                loss, components = objective(head, batch, group, temperature(step))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite hardening loss")
                loss.backward()
                norm = float(torch.stack([p.grad.detach().double().square().sum()
                                          for p in head.parameters() if p.grad is not None]).sum().sqrt())
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError("inactive or nonfinite hardening gradient")
                max_gradient = max(max_gradient, norm)
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(head)
            if step <= 2 or step % 100 == 0 or step == steps:
                with budget.cuda(), torch.no_grad():
                    _, ema_values = objective(ema.shadow, batch, group, temperature(step))
                    _, ema_fixed = objective(ema.shadow, batch, group, 0.04)
                row = {"step": step, "tau": temperature(step), "live": components,
                       "gradient_norm_before_clip": norm, "ema_same_batch_after_update": ema_values,
                       "ema_fixed_tau_0_04": ema_fixed, "learning_rate_after_update": scheduler.get_last_lr()[0]}
                history.append(row)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                print(json.dumps({"stage": "hardening_train", **row}, sort_keys=True), flush=True)
    endpoint = ema.shadow.eval().requires_grad_(False)
    return endpoint, {"steps": steps, "seed": SEED, "batch_size": BATCH, "batch_sha256": batch_hash.hexdigest(),
                      "initial_state_sha256": state_hash(initial), "endpoint_state_sha256": state_hash(endpoint),
                      "max_gradient_norm": max_gradient, "history": history,
                      "optimizer": {"kind": "AdamW", "lr": 4e-4, "weight_decay": 1e-4, "clip": 1.0,
                                    "cosine_T_max": 2000, "min_lr": 1e-5, "ema_decay": 0.9995},
                      "selection": "fixed_final_EMA_only", "seconds": time.monotonic() - started}


def continue_head(point_path, head_path, config_path, output_dir, smoke=False):
    """Freshly cache released clean81 and train one fixed head; never load a router.

    Caller supplies newly generated scratch components or explicitly authorized
    development components. Paths are never replaced by historical fallbacks.
    Returns the same report stored at output_dir/report.json.
    """
    point_path, head_path, config_path, output = map(Path, (point_path, head_path, config_path, output_dir))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite hardening output: {output}")
    sources = {"point": point_path, "head": head_path, "config": config_path, "module": Path(__file__)}
    hashes_before = {name: file_hash(path) for name, path in sources.items()}
    config = load_config(config_path)
    settings = config["fixed_decision"]
    if settings["center_multiplier"] != 1.5 or settings["width_multiplier"] != 1.0:
        raise RuntimeError("fixed deployment constants differ")
    device = choose_device(config.get("device", "auto"))
    if not smoke and device.type != "cuda":
        raise RuntimeError("full hardening requires CUDA")
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 4:
        torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(SEED)
    cases = discover_cases(config["data"]["real_dir"], exclude_files=config["data"]["exclude_files"])
    dataset = Native79Dataset(cases)
    if (len(cases), len(dataset)) != (81, 3179) or "7575_0.h5" in [case.path.name for case in cases]:
        raise RuntimeError("released clean81/3179 Native79 inventory differs")
    point_payload = torch.load(point_path, map_location="cpu", weights_only=False)
    head_payload = torch.load(head_path, map_location="cpu", weights_only=False)
    if "decision_settings" in head_payload and head_payload["decision_settings"] != settings:
        raise RuntimeError("input head deployment constants differ from config")
    point = HybridFlowOperator(ModelConfig.from_dict(point_payload["model_config"])).to(device)
    point.load_state_dict(point_payload["state_dict"], strict=True)
    point.eval().requires_grad_(False)
    initial = FastIntervalHead(FastIntervalConfig.from_dict(head_payload["model_config"])).to(device)
    initial.load_state_dict(head_payload["state_dict"], strict=True)
    initial.eval().requires_grad_(False)
    point_before, initial_before = state_hash(point), state_hash(initial)
    group = torch.tensor(settings["group_width_multipliers"], device=device, dtype=torch.float32)[None, :, None, None, :]
    if group.shape != (1, 20, 1, 1, 2):
        raise RuntimeError("fixed group width shape differs")
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output, device)
    cache, cache_report = training_cache(point, dataset, device, 16 if smoke else 3179, budget)
    endpoint, training = train_endpoint(initial, cache, group, device, output, budget, 2 if smoke else STEPS)
    del cache
    exported = copy.deepcopy(head_payload)
    exported["state_dict"] = {name: value.cpu() for name, value in endpoint.state_dict().items()}
    exported["decision_settings"] = settings
    torch.save(exported, output / "final_head.pt")
    validity = {"point_state_exact": state_hash(point) == point_before,
                "original_head_state_exact": state_hash(initial) == initial_before,
                "sources_unchanged": hashes_before == {name: file_hash(path) for name, path in sources.items()},
                "point_has_no_gradients": all(p.grad is None and not p.requires_grad for p in point.parameters()),
                "head_changed": state_hash(endpoint) != initial_before,
                "head_finite": all(bool(torch.isfinite(v).all()) for v in endpoint.state_dict().values()),
                "cache_fp32_exact_count": cache_report["rows"] == (16 if smoke else 3179) and cache_report["dtype"] == "float32",
                "fixed_steps": training["steps"] == (2 if smoke else 2000),
                "positive_gradient": training["max_gradient_norm"] > 0,
                "no_router_in_training": True, "no_checkpoint_selection": True}
    budget.check()
    passed = all(validity.values())
    report = {"status": ("SMOKE_PASS" if smoke else "FULL_COMPLETE") if passed else "INVALID",
              "mode": "smoke" if smoke else "full", "validity_pass": passed, "validity": validity,
              "source_hashes": hashes_before, "point_state_before": point_before, "point_state_after": state_hash(point),
              "source_head_state_sha256": initial_before, "endpoint_state_sha256": state_hash(endpoint),
              "artifact_hashes": {"interval_head": file_hash(output / "final_head.pt"),
                                  "training_curve": file_hash(output / "training_curve.jsonl")},
              "training_cache": cache_report, "training": training, "decision_settings": settings,
              "released_data": {"cases": len(cases), "native79_windows": len(dataset),
                                "inventory_sha256": hashlib.sha256("\n".join(f"{c.path.name}|{c.reynolds:g}|{c.angle:g}|{c.frames}" for c in cases).encode()).hexdigest(),
                                "fit_regimes": [list(c.regime) for c in cases], "excluded_files": config["data"]["exclude_files"]},
              "runtime": {"wall_seconds": time.monotonic() - budget.started, "cuda_seconds": budget.gpu_seconds},
              "selection": "fixed_final_EMA_only", "performance_claim": None,
              "smoke_is_not_submittable": bool(smoke), "network_or_router_loaded": False}
    write_json(output / "report.json", report)
    if not passed:
        raise RuntimeError("hardening validity failed; report preserved")
    return report
