"""Fixed additive radius refinement, using training data and explicit checkpoints only."""
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
from .sps_radius_capacity import ExpandedFastIntervalHead, load_expanded_head


def refine_capacity(point, center, radius, cache, output, budget, device):
    expanded = ExpandedFastIntervalHead.expand(radius, seed=20260907579).to(device)
    expanded.eval().requires_grad_(False)
    with budget.cuda(), torch.no_grad():
        diagnostics = cache['diagnostics'][:16].to(device)
        expected, actual = radius(diagnostics), expanded(diagnostics)
        if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
            raise RuntimeError('initial function mismatch')
    record = refine_head(point, center, expanded, cache, 'radius', output, budget, device)
    for endpoint in ('live', 'ema_audit_only'):
        source = output / ('radius_' + endpoint + '.pt')
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        checkpoint['architecture'] = ExpandedFastIntervalHead.architecture
        checkpoint['metadata']['capacity_seed'] = 20260907579
        path = output / ('capacity_' + endpoint + '.pt')
        torch.save(checkpoint, path)
        restored = load_expanded_head(path, device)
        if method.state_hash(restored) != record['endpoint_states'][endpoint]:
            raise RuntimeError('serialized capacity state mismatch')
    record['initial_function_bitexact'] = True
    record['capacity_seed'] = 20260907579
    record['architecture'] = ExpandedFastIntervalHead.architecture
    _write(output / 'training_report.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('point', 'center', 'radius', 'real-dir', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-cache', type=Path)
    parser.add_argument('--expected-state')
    parser.add_argument('--gpu-seconds', type=float, default=900)
    parser.add_argument('--wall-seconds', type=float, default=1200)
    args = parser.parse_args(argv)
    if args.gpu_seconds <= 0 or args.wall_seconds <= 0:
        parser.error('budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    budget = TrainingBudget(args.output, args.gpu_seconds, args.wall_seconds)
    sources = {name: getattr(args, name).resolve() for name in ('point', 'center', 'radius')}
    pins = {str(p): method.file_hash(p) for p in sources.values()}
    names = json.loads(args.cases.read_text())
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError('cases must be unique names')
    inventory = {c.path.name: c for c in discover_cases(args.real_dir, exclude_files=['7575_0.h5'])}
    if set(names) - inventory.keys():
        raise ValueError('case not present in allowed release inventory')
    cases = [inventory[n] for n in names]
    for case in cases:
        pins[str(case.path.resolve())] = method.file_hash(case.path)
    _write(args.output / 'job.json', {'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': list(os.sys.argv), 'source_pins': pins, 'cases': names})
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
        budget.gpu_started = time.monotonic()
        point = method.load_point(sources['point'], device)
        center = method.load_head(sources['center'], device)
        radius = method.load_head(sources['radius'], device)
        dataset = method.Native79Dataset(cases)
        cache, record = method.training_cache(point, dataset, device, len(dataset), budget, max_cache_bytes=8 * 1024**3)
        _write(args.output / 'training_cache.json', record)
        if args.expected_cache:
            expected = json.loads(args.expected_cache.read_text())
            for key, value in expected.items():
                if key != 'seconds' and record[key] != value:
                    raise RuntimeError('cache mismatch: ' + key)
        result = refine_capacity(point, center, radius, cache, args.output, budget, device)
        if args.expected_state and result['endpoint_states']['live'] != args.expected_state:
            raise RuntimeError('fixed endpoint state does not reproduce')
        for path, digest in pins.items():
            if method.file_hash(path) != digest:
                raise RuntimeError('source changed: ' + path)
        _write(args.output / 'report.json', {'status': 'COMPLETE', 'validity': 'VALID', 'endpoint_states': result['endpoint_states'], 'expected_state_exact': args.expected_state is not None, 'source_pins_unchanged': True, 'training_windows': len(dataset), 'evaluation_arrays_read': False, 'budget': budget.check()})
    except BaseException as error:
        _write(args.output / 'failure.json', {'status': 'DEFERRED_BY_RESOURCE' if 'RESOURCE_CAP' in str(error) else 'INVALID', 'exception': repr(error), 'budget': budget.snapshot()})
        raise
