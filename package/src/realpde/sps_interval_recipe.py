"""Fixed four-stage SPS refinement of a frozen point and initial joint head.

Inputs are explicit checkpoints and released training cases. No experiment,
held data, legacy deployment constants, or validation-selected endpoints are used.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import torch
from . import identity_interval_hardening as method
from .data import discover_cases
from .fixed_sps_head_training import TrainingBudget, refine_head, _write
from .fixed_sps_capacity_training import refine_capacity

STAGES = ('baseline_reward', 'sps_center', 'radius_refinement', 'radius_capacity')


def run_recipe(point, head, cache, output, budget, device, expected_states=None):
    records = {}
    center, radius = head, head
    for stage in STAGES:
        destination = output / stage
        destination.mkdir(exist_ok=False)
        if stage == 'radius_capacity':
            result = refine_capacity(point, center, radius, cache, destination, budget, device)
        else:
            mode = 'center' if stage == 'sps_center' else 'radius'
            result = refine_head(point, center, radius, cache, mode, destination, budget, device)
            _write(destination / 'training_report.json', result)
            updated = method.load_head(destination / (mode + '_live.pt'), device)
            if mode == 'center':
                center = updated
            else:
                radius = updated
        records[stage] = result
        _write(output / 'stage_records.json', records)
        if expected_states and result['endpoint_states']['live'] != expected_states[stage]:
            raise RuntimeError('fixed state mismatch: ' + stage)
        print(json.dumps({'stage': stage, 'state': result['endpoint_states']['live'], **budget.check()}), flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('point', 'joint-head', 'real-dir', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-cache', type=Path)
    parser.add_argument('--expected-states', type=Path)
    parser.add_argument('--gpu-seconds', type=float, default=900)
    parser.add_argument('--wall-seconds', type=float, default=1200)
    args = parser.parse_args(argv)
    if args.gpu_seconds <= 0 or args.wall_seconds <= 0:
        parser.error('budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    budget = TrainingBudget(args.output, args.gpu_seconds, args.wall_seconds)
    names = json.loads(args.cases.read_text())
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError('cases must be unique names')
    inventory = {c.path.name: c for c in discover_cases(args.real_dir, exclude_files=['7575_0.h5'])}
    if set(names) - inventory.keys():
        raise ValueError('case absent from allowed inventory')
    cases = [inventory[n] for n in names]
    paths = [args.point.resolve(), args.joint_head.resolve(), args.cases.resolve(), *[c.path.resolve() for c in cases]]
    pins = {str(p): method.file_hash(p) for p in paths}
    _write(args.output / 'job.json', {'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': list(os.sys.argv), 'source_pins': pins, 'cases': names})
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
        budget.gpu_started = time.monotonic()
        point = method.load_point(args.point, device)
        head = method.load_head(args.joint_head, device)
        states = {'point': method.state_hash(point), 'head': method.state_hash(head)}
        dataset = method.Native79Dataset(cases)
        cache, record = method.training_cache(point, dataset, device, len(dataset), budget, max_cache_bytes=8 * 1024**3)
        _write(args.output / 'training_cache.json', record)
        if args.expected_cache:
            expected = json.loads(args.expected_cache.read_text())
            for key, value in expected.items():
                if key != 'seconds' and record[key] != value:
                    raise RuntimeError('cache mismatch: ' + key)
        expected_states = json.loads(args.expected_states.read_text()) if args.expected_states else None
        if expected_states is not None and set(expected_states) != set(STAGES):
            raise ValueError('expected states must cover exactly four stages')
        records = run_recipe(point, head, cache, args.output, budget, device, expected_states)
        for name, model in [('point', point), ('head', head)]:
            if method.state_hash(model) != states[name]:
                raise RuntimeError('frozen source changed')
        for path, digest in pins.items():
            if method.file_hash(path) != digest:
                raise RuntimeError('input changed: ' + path)
        _write(args.output / 'report.json', {'status': 'COMPLETE', 'validity': 'VALID', 'steps': 8000, 'stages': {k: v['endpoint_states'] for k, v in records.items()}, 'all_expected_states_exact': expected_states is not None, 'source_pins_unchanged': True, 'training_windows': len(dataset), 'evaluation_arrays_read': False, 'budget': budget.check()})
    except BaseException as error:
        _write(args.output / 'failure.json', {'status': 'DEFERRED_BY_RESOURCE' if 'RESOURCE_CAP' in str(error) else 'INVALID', 'exception': repr(error), 'budget': budget.snapshot()})
        raise
