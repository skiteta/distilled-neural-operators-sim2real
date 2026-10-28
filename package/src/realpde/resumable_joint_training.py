"""Fixed joint training with atomic checkpoints and exact data/RNG continuation.

Optimizer/loss/update order is the preserved fixed joint recipe. A checkpoint
is saved only after both optimizers, schedulers and EMAs finish a full step.
Budget exceptions are handled before the next batch. Abrupt termination may
lose work since the latest completed checkpoint, never produce a partial step.
"""
from __future__ import annotations
import copy, hashlib, json, os, random, time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
from .dense_start_sps_recipe import (file_hash,state_hash,all_finite,load_point,
    joint_interval_objective,HybridFlowOperator,FastIntervalHead,CaseInfo,
    MetricAwareLoss,LossWeights,ExponentialMovingAverage,seed_everything,TrainingBudget)

class PlannedPause(RuntimeError):
    """Controlled checkpoint boundary used by the exact-resume smoke test."""

class ResumableBatches:
    """Replay only integer indices to restore a shuffled, zero-worker loader.

    Matches DataLoader's dedicated-generator base-seed and RandomSampler
    consumption exactly. Skipped batches never read trajectories or consume
    model/global RNG. The real datasets in this recipe are deterministic.
    """
    def __init__(self, dataset, *, batch_size, seed):
        self.dataset=dataset
        self.generator=torch.Generator().manual_seed(int(seed))
        self.loader=DataLoader(range(len(dataset)), batch_size=batch_size, shuffle=True,
                               generator=self.generator, num_workers=0,
                               drop_last=len(dataset)>=batch_size)
        self.iterator=None
        self.epoch_state=None
        self.offset=0
    def _start_epoch(self):
        self.epoch_state=self.generator.get_state().clone()
        self.iterator=iter(self.loader)
        self.offset=0
    def __iter__(self): return self
    def __next__(self):
        if self.iterator is None: self._start_epoch()
        try: indices=next(self.iterator)
        except StopIteration:
            self._start_epoch()
            indices=next(self.iterator)
        batch=default_collate([self.dataset[int(i)] for i in indices])
        self.offset+=1
        return batch
    def state_dict(self):
        return dict(epoch_state=self.epoch_state, offset=self.offset,
                    generator_state=self.generator.get_state().clone())
    def load_state_dict(self, state):
        self.iterator=None
        self.offset=0
        self.epoch_state=None
        if state['epoch_state'] is None:
            self.generator.set_state(state['generator_state'])
            return
        self.generator.set_state(state['epoch_state'])
        self._start_epoch()
        for _ in range(state['offset']): next(self.iterator)
        self.offset=state['offset']
        if not torch.equal(self.generator.get_state(), state['generator_state']):
            raise ValueError('Sampler RNG reconstruction mismatch')

def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])

def atomic_torch_save(payload, path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def train_joint(
    source: Path,
    source_head: FastIntervalHead,
    cases: Sequence[CaseInfo],
    config: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
    dataset: Dataset,
    budget: TrainingBudget,
    checkpoint_dir: Path,
    resume: Path | None = None,
    checkpoint_every: int = 250,
    stop_after: int | None = None,
    progress_callback=None,
) -> tuple[HybridFlowOperator, FastIntervalHead, dict[str, Any], Dataset]:
    settings = config["training"]["joint_fullfit"]
    seed = int(settings["seed"])
    seed_everything(seed)
    model = load_point(source, device).train().requires_grad_(True)
    head = copy.deepcopy(source_head).to(device).train().requires_grad_(True)
    if [c.path.name for c in dataset.cases] != [c.path.name for c in cases]:
        raise ValueError("dataset cases mismatch")
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
    if int(config["data"]["num_workers"]) != 0:
        raise ValueError("Exact resume requires num_workers=0")
    batches = ResumableBatches(dataset, batch_size=int(settings["batch_size"]), seed=seed + 202)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha256=file_hash(source), initial_head_state=state_hash(source_head),
                    config=config, total_steps=int(steps), dataset_length=len(dataset),
                    cases=[c.path.name for c in cases],
                    windows_sha256=hashlib.sha256(json.dumps(dataset.windows).encode()).hexdigest())
    modules = dict(point=model, head=head, point_ema=point_ema.shadow, head_ema=head_ema.shadow)
    optimizers = dict(point=point_optimizer, head=head_optimizer)
    schedulers = dict(point=point_scheduler, head=head_scheduler)
    completed = 0
    prior_last = {}
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        for name, value in modules.items(): value.load_state_dict(payload["modules"][name], strict=True)
        for name, value in optimizers.items(): value.load_state_dict(payload["optimizers"][name])
        for name, value in schedulers.items(): value.load_state_dict(payload["schedulers"][name])
        batches.load_state_dict(payload["batches"])
        restore_rng(payload["rng"])
        completed = int(payload["completed_steps"])
        prior_last = payload["last"]
    def save(completed_steps, last):
        path = checkpoint_dir / ("step_%06d.pt" % completed_steps)
        if path.exists():
            raise FileExistsError("Refuse to overwrite checkpoint: " + str(path))
        payload = dict(schema=1, identity=identity, completed_steps=completed_steps, last=last,
                       modules={k:v.state_dict() for k,v in modules.items()},
                       optimizers={k:v.state_dict() for k,v in optimizers.items()},
                       schedulers={k:v.state_dict() for k,v in schedulers.items()},
                       batches=batches.state_dict(), rng=capture_rng(), budget_at_save=budget.snapshot())
        atomic_torch_save(payload, path)
        pointer=checkpoint_dir / "latest.json"
        temporary=pointer.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(path=path.name, completed_steps=completed_steps,
                                              sha256=file_hash(path))) + "\n")
        os.replace(temporary, pointer)
        return path
    started = time.perf_counter()
    last: dict[str, Any] = prior_last
    last_saved = completed if resume is not None else -1
    for step in range(completed + 1, int(steps) + 1):
        try:
            budget.check()
        except BaseException:
            if last_saved != step - 1:
                save(step - 1, last)
            raise
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
            if progress_callback is not None: progress_callback(last)
        if step % checkpoint_every == 0 or step == int(steps) or step == stop_after:
            save(step, last)
            last_saved = step
        if step == stop_after and step < int(steps):
            raise PlannedPause("Saved completed step " + str(step))
    point_final = point_ema.shadow.eval().requires_grad_(False)
    head_final = head_ema.shadow.eval().requires_grad_(False)
    return point_final, head_final, {
        "stage": "joint_fullfit",
        "steps": int(steps),
        "resumed_from_step": completed,
        "checkpoint_dir": str(checkpoint_dir),
        "seed": seed,
        "training_windows": len(dataset),
        "runtime_seconds": time.perf_counter() - started,
        "final": last,
        "point_state_hash": state_hash(point_final),
        "head_state_hash": state_hash(head_final),
        "point_all_finite": all_finite(point_final),
        "head_all_finite": all_finite(head_final),
    }, dataset
