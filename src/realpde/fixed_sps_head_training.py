"""Fixed 2,000-step SPS head refinement using released training cases only.

The point model is frozen. Center refinement freezes a separate radius head;
radius refinement freezes a separate center head. This module has no experiment,
held-set, leaderboard, calibration, or submission dependency.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from . import identity_interval_hardening as method
from .data import discover_cases


class TrainingBudget:
    def __init__(self, output: Path, gpu_seconds: float, wall_seconds: float):
        self.output = output
        self.started = time.monotonic()
        self.gpu_started = None
        self.gpu_limit = gpu_seconds
        self.wall_limit = wall_seconds

    def snapshot(self):
        now = time.monotonic()
        return {
            "wall_seconds": now - self.started,
            "gpu_reservation_seconds": 0 if self.gpu_started is None else now - self.gpu_started,
            "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_device_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            ),
        }

    def check(self):
        values = self.snapshot()
        if values["wall_seconds"] > self.wall_limit:
            raise RuntimeError("RESOURCE_CAP wall_seconds")
        if values["gpu_reservation_seconds"] > self.gpu_limit:
            raise RuntimeError("RESOURCE_CAP gpu_reservation_seconds")
        if values["peak_host_bytes"] > 12 * 1024**3:
            raise RuntimeError("RESOURCE_CAP host memory")
        if values["peak_device_bytes"] > 8 * 1024**3:
            raise RuntimeError("RESOURCE_CAP device memory")
        return values

    @contextlib.contextmanager
    def cuda(self):
        if self.gpu_started is None:
            self.gpu_started = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.check()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def refine_head(point, center, radius, cache, mode, output, budget, device):
    """Run one fixed schedule. LIVE is the endpoint; EMA is saved only for audit."""
    if mode not in {"center", "radius"}:
        raise ValueError("mode must be center or radius")
    initial = center if mode == "center" else radius
    head = copy.deepcopy(initial).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=2000, eta_min=1e-5
    )
    ema = method.ExponentialMovingAverage(head, .9995)
    generator = torch.Generator().manual_seed(2026090541)
    batches = hashlib.sha256()
    frozen = {name: method.state_hash(model) for name, model in (
        ("point", point), ("center", center), ("radius", radius)
    )}
    history = []
    with (output / "training_curve.jsonl").open("x") as stream:
        for step in range(1, 2001):
            budget.check()
            indices = torch.randint(len(cache["point"]), (8,), generator=generator)
            batches.update(indices.numpy().tobytes())
            with budget.cuda():
                batch = {key: value.index_select(0, indices).to(device)
                         for key, value in cache.items()}
                with torch.no_grad():
                    factor = method.sample_factor(batch["point"], batch["target"])
                    if mode == "center":
                        _, fixed_radius = radius(batch["diagnostics"])
                        width_factor = torch.exp(
                            -2 * fixed_radius * batch["scale"] / method.SIGMA_GLOBAL
                        )
                    else:
                        fixed_shift, _ = center(batch["diagnostics"])
                measured = batch["target"] != 0
                error = (batch["target"] - batch["point"]) / batch["scale"]
                tau = method.temperature(step)
                optimizer.zero_grad(set_to_none=True)
                shift, predicted_radius = head(batch["diagnostics"])
                if mode == "center":
                    reward = (
                        factor[:, None, None, None, None]
                        * torch.sigmoid((fixed_radius - (error - shift).abs()) / tau)
                        * width_factor
                    )
                else:
                    reward = (
                        factor[:, None, None, None, None]
                        * torch.sigmoid((predicted_radius - (error - fixed_shift).abs()) / tau)
                        * torch.exp(-2 * predicted_radius * batch["scale"] / method.SIGMA_GLOBAL)
                    )
                loss = -.75 * reward[measured].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                norm = method.gradient_norm(head)
                if not np.isfinite(norm) or norm <= 0:
                    raise FloatingPointError("nonfinite or inactive gradient")
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                ema.update(head)
                row = {"step": step, "tau": tau, "loss": float(loss.detach()),
                       "gradient_norm": norm, "lr": scheduler.get_last_lr()[0]}
            if step <= 2 or step % 100 == 0:
                history.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({"stage": "training", **row, **budget.check()}), flush=True)
    for name, model in (("point", point), ("center", center), ("radius", radius)):
        if method.state_hash(model) != frozen[name]:
            raise RuntimeError("frozen model changed: " + name)
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("gradient reached frozen model: " + name)
    endpoints = {"live": head, "ema_audit_only": ema.shadow}
    states = {}
    for name, model in endpoints.items():
        model.eval().requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        if not method.all_finite(model):
            raise FloatingPointError("nonfinite endpoint")
        states[name] = method.state_hash(model)
        torch.save({"model_config": model.config.to_dict(), "state_dict": model.state_dict(),
                    "metadata": {"mode": mode, "steps": 2000, "endpoint": name,
                                 "selection": "fixed_final_no_validation"}},
                   output / (mode + "_" + name + ".pt"))
    return {"steps": 2000, "seed": 2026090541, "batch_size": 8,
            "batches_sha256": batches.hexdigest(), "frozen_states": frozen,
            "endpoint_states": states, "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("center", "radius"), required=True)
    for name in ("point", "center", "radius", "real-dir", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-cache", type=Path)
    parser.add_argument("--expected-state")
    parser.add_argument("--continue-radius", action="store_true",
                        help="After reproducing center, perform one fixed radius refinement")
    parser.add_argument("--gpu-seconds", type=float, default=600)
    parser.add_argument("--wall-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    if args.continue_radius and args.mode != "center":
        parser.error("--continue-radius requires --mode center")
    if args.gpu_seconds <= 0 or args.wall_seconds <= 0:
        parser.error("budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    budget = TrainingBudget(args.output, args.gpu_seconds, args.wall_seconds)
    source_paths = {name: getattr(args, name).resolve() for name in ("point", "center", "radius")}
    source_pins = {str(path): method.file_hash(path) for path in source_paths.values()}
    names = json.loads(args.cases.read_text())
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError("cases must be a JSON array of unique filenames")
    inventory = {case.path.name: case for case in discover_cases(
        args.real_dir, exclude_files=["7575_0.h5"]
    )}
    if set(names) - inventory.keys():
        raise ValueError("requested cases absent from allowed released inventory")
    cases = [inventory[name] for name in names]
    for case in cases:
        source_pins[str(case.path.resolve())] = method.file_hash(case.path)
    _write(args.output / "job.json", {"pid": os.getpid(), "started_at":
           datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "command": list(os.sys.argv), "source_pins": source_pins, "cases": names})
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        budget.gpu_started = time.monotonic()
        point = method.load_point(source_paths["point"], device)
        center = method.load_head(source_paths["center"], device)
        radius = method.load_head(source_paths["radius"], device)
        dataset = method.Native79Dataset(cases)
        cache, record = method.training_cache(
            point, dataset, device, len(dataset), budget, max_cache_bytes=8 * 1024**3
        )
        _write(args.output / "training_cache.json", record)
        if args.expected_cache:
            expected = json.loads(args.expected_cache.read_text())
            for key, value in expected.items():
                if key != "seconds" and record[key] != value:
                    raise RuntimeError("cache mismatch: " + key)
        result = refine_head(point, center, radius, cache, args.mode, args.output, budget, device)
        _write(args.output / "training_report.json", result)
        if args.expected_state and result["endpoint_states"]["live"] != args.expected_state:
            raise RuntimeError("endpoint state does not reproduce expected state")
        center_reproduction = None
        if args.continue_radius:
            center_reproduction = result
            center = method.load_head(args.output / "center_live.pt", device)
            radius_output = args.output / "radius_refinement"
            radius_output.mkdir(exist_ok=False)
            result = refine_head(point, center, radius, cache, "radius", radius_output,
                                 budget, device)
            _write(radius_output / "training_report.json", result)
        for path, digest in source_pins.items():
            if method.file_hash(path) != digest:
                raise RuntimeError("input changed during fitting: " + path)
        _write(args.output / "report.json", {"status": "COMPLETE", "validity": "VALID",
               "continued_radius_once": args.continue_radius,
               "center_reproduction_states": None if center_reproduction is None else
                   center_reproduction["endpoint_states"],
               "mode": args.mode, "endpoint_states": result["endpoint_states"],
               "expected_state_exact": args.expected_state is not None,
               "source_pins_unchanged": True, "training_windows": len(dataset),
               "evaluation_arrays_read": False, "budget": budget.check()})
    except BaseException as error:
        _write(args.output / "failure.json", {"exception": repr(error),
               "status": "DEFERRED_BY_RESOURCE" if "RESOURCE_CAP" in str(error) else "INVALID",
               "budget": budget.snapshot()})
        raise
