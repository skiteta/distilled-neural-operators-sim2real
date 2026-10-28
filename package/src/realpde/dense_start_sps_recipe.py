"""Portable fixed dense-start joint and SPS interval recipe from explicit inputs.

Reuses released source weights and training cases; no experiment imports or
validation-based decisions. The starting sim and initial joint head are explicit
intermediate artifacts, not a claim of rerunning their upstream training.
"""
from __future__ import annotations
import argparse,copy,datetime,gc,hashlib,json,os,time
from pathlib import Path
from typing import Any,Iterator,Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from .config import load_config
from .data import CaseInfo,discover_cases
from .fast_interval import FastIntervalHead
from .flow_diagnostics import flow_diagnostics
from .losses import LossWeights,MetricAwareLoss
from .metrics import SIGMA_GLOBAL,_mvpe_relative_l2_per_sample,_relative_l2_per_sample,_tke_relative_l2_per_sample
from .model import HybridFlowOperator,ModelConfig
from .train import ExponentialMovingAverage,seed_everything
from . import identity_interval_hardening as method
from .fixed_sps_head_training import TrainingBudget,_write
from .temporal_start_dataset import ReleasedTemporalStartDataset
from .sps_interval_recipe import run_recipe
from .sps_radius_capacity import load_expanded_head
from .fixed_hidden_sps_training import hidden_training_cache,train_hidden_residual

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

def load_point(path: Path, device: torch.device) -> HybridFlowOperator:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HybridFlowOperator(ModelConfig.from_dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model

def sample_factor(point: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = _relative_l2_per_sample(point, target)
    tke = _tke_relative_l2_per_sample(point, target)
    mvpe = _mvpe_relative_l2_per_sample(point, target)
    return (
        0.5 * (1.0 - rel / (0.5 + rel))
        + 0.3 * (1.0 - tke / (0.5 + tke))
        + 0.2 * (1.0 - mvpe / (0.5 + mvpe))
    )

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
    dataset: Dataset,
    budget: TrainingBudget,
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
        budget.check()
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


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('sim-point','initial-head','config','real-dir','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--expected-states',type=Path)
 parser.add_argument('--joint-only',action='store_true',help='Run and verify only fixed8000point/joint-head training; do not claim full downstream reproduction')
 parser.add_argument('--expected-native-cache',type=Path)
 parser.add_argument('--expected-hidden-cache',type=Path)
 parser.add_argument('--gpu-seconds',type=float,default=1800)
 parser.add_argument('--wall-seconds',type=float,default=2400)
 args=parser.parse_args(argv)
 if min(args.gpu_seconds,args.wall_seconds)<=0:parser.error('budget must be positive')
 args.output.mkdir(parents=True,exist_ok=False);budget=TrainingBudget(args.output,args.gpu_seconds,args.wall_seconds);names=json.loads(args.cases.read_text())
 if not isinstance(names,list) or len(names)!=len(set(names)):raise ValueError('case list must have unique names')
 inventory={c.path.name:c for c in discover_cases(args.real_dir,exclude_files=['7575_0.h5'])}
 if set(names)-inventory.keys():raise ValueError('case absent from allowed release inventory')
 cases=[inventory[n] for n in names];paths=[args.sim_point.resolve(),args.initial_head.resolve(),args.config.resolve(),args.cases.resolve(),*[c.path.resolve() for c in cases]];pins={str(p):file_hash(p) for p in paths};cfg=load_config(args.config);cfg.pop('fixed_decision',None);cfg['data']['num_workers']=0
 expected={} if args.expected_states is None else json.loads(args.expected_states.read_text())
 _write(args.output/'job.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=list(os.sys.argv),source_pins=pins,cases=names))
 try:
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda');budget.gpu_started=time.monotonic();dataset=ReleasedTemporalStartDataset(cases,stride=1,budget=budget);source_head=method.load_head(args.initial_head,device)
  point,head,joint_record,_=train_joint(args.sim_point,source_head,cases,cfg,device,steps=8000,dataset=dataset,budget=budget);del dataset,source_head,_;gc.collect();torch.cuda.empty_cache()
  for name,model in [('point',point),('joint_head',head)]:
   if expected and state_hash(model)!=expected[name]:raise RuntimeError('dense joint state mismatch: '+name)
   torch.save(dict(model_config=model.config.to_dict(),state_dict=model.state_dict(),metadata=dict(recipe='dense_start_joint_fixed8000',endpoint='EMA.9995')),args.output/(name+'.pt'))
  _write(args.output/'joint_training_report.json',joint_record)
  if args.joint_only:
   for path,digest in pins.items():
    if file_hash(path)!=digest:raise RuntimeError('input changed: '+path)
   _write(args.output/'report.json',dict(status='COMPLETE',validity='VALID',scope='JOINT_ONLY',point_state=state_hash(point),joint_head_state=state_hash(head),all_expected_joint_states_exact=bool(expected),source_pins_unchanged=True,joint_steps=8000,interval_steps=0,full_downstream_reproduction=False,evaluation_arrays_read=False,budget=budget.check()))
   return
  native=method.Native79Dataset(cases);cache,record=method.training_cache(point,native,device,len(native),budget,max_cache_bytes=8*1024**3);_write(args.output/'native_training_cache.json',record)
  if args.expected_native_cache:
   old=json.loads(args.expected_native_cache.read_text())
   for key,value in old.items():
    if key!='seconds' and record[key]!=value:raise RuntimeError('native cache mismatch: '+key)
  output=args.output/'sps_recipe';output.mkdir();stages=run_recipe(point,head,cache,output,budget,device,expected.get('sps_stages'));del cache;gc.collect();torch.cuda.empty_cache();center=method.load_head(output/'sps_center/center_live.pt',device);radius=load_expanded_head(output/'radius_capacity/capacity_live.pt',device);frozen=dict(point=point,center=center,radius=radius)
  output=args.output/'hidden_recipe';output.mkdir();old=None if args.expected_hidden_cache is None else json.loads(args.expected_hidden_cache.read_text());cache=hidden_training_cache(frozen,native,device,output,budget,old);hidden=train_hidden_residual(frozen,cache,device,output,budget)['aligned']
  if expected and state_hash(hidden)!=expected['hidden']:raise RuntimeError('hidden state mismatch')
  for path,digest in pins.items():
   if file_hash(path)!=digest:raise RuntimeError('input changed: '+path)
  _write(args.output/'report.json',dict(status='COMPLETE',validity='VALID',point_state=state_hash(point),joint_head_state=state_hash(head),sps_stages={k:v['endpoint_states']['live'] for k,v in stages.items()},hidden_state=state_hash(hidden),all_expected_states_exact=bool(expected),source_pins_unchanged=True,joint_steps=8000,interval_steps=12000,evaluation_arrays_read=False,budget=budget.check()))
 except BaseException as error:
  _write(args.output/'failure.json',dict(status='DEFERRED_BY_RESOURCE' if 'RESOURCE_CAP' in str(error) else 'INVALID',exception=repr(error),budget=budget.snapshot()));raise
