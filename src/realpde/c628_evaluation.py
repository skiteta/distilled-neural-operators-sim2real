"""Evaluate a completed training plan, with fresh processes per case and arm.

Cold and warm timings are separate local measurements, not a claim about the
unverified official warm-up boundary. No outer target is used by training.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from .c628_pipeline import ROOT, read, write, validate, check_sources, digest


def verify_completion(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    validate(plan)
    check_sources(plan)
    out = Path(plan['output'])
    if out != plan_path.parent:
        raise ValueError('bound output mismatch')
    pin = digest(plan_path)
    for node in plan['nodes']:
        seal = read(out / node['id'] / 'complete.json')
        if seal['plan_sha256'] != pin or any(digest(out / p) != h for p, h in seal['outputs'].items()):
            raise ValueError('invalid completed training stage: ' + node['id'])
    return plan


def aggregate(rows):
    import math
    n = sum(r['windows'] for r in rows)
    count = sum(r['scored_elements'] for r in rows)
    if not n or not count:
        raise ValueError('empty evaluation')
    result = {}
    for key in ('rel_l2', 'tke', 'mvpe'):
        error = sum(r['mean_errors'][key] * r['windows'] for r in rows) / n
        result[key + '_score'] = 100 / (1 + .5 * error)
    result['sps_score'] = sum(r['sps_score'] * r['scored_elements'] for r in rows) / count
    result['coverage'] = sum(r['coverage'] * r['scored_elements'] for r in rows) / count
    for mode in ('cold', 'warm'):
        result[mode + '_seconds_per_window'] = sum(r[mode + '_seconds'] for r in rows) / n
    result.update(windows=n, cases=len(rows), scored_elements=count)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite aggregate')
    return result


def paired_case_summary(rows):
    import numpy as np
    by_arm = {a: {r['case']: r for r in values} for a, values in rows.items()}
    if set(by_arm['joint_identity']) != set(by_arm['c628']):
        raise ValueError('paired cases differ')
    names = sorted(by_arm['c628'])
    deltas = np.array([by_arm['c628'][n]['sps_score'] - by_arm['joint_identity'][n]['sps_score'] for n in names])
    rng = np.random.default_rng(20260911675)
    means = deltas[rng.integers(len(names), size=(10000, len(names)))].mean(1)
    slices = {}
    for n, delta in zip(names, deltas):
        angle = n.removesuffix('.h5').split('_')[1]
        slices.setdefault(angle, []).append(float(delta))
    return {'SPS_delta_equal_case_mean': float(deltas.mean()),
            'paired_case90': np.quantile(means, [.05, .95]).tolist(),
            'case_deltas': dict(zip(names, map(float, deltas))),
            'AoA_equal_case_deltas': {a: float(np.mean(v)) for a, v in slices.items()},
            'scope': 'Paired case resampling on historically exposed release; not a fresh Private confidence interval.'}


def evaluate(plan_path, max_wall_seconds):
    if max_wall_seconds <= 0:
        raise ValueError('explicit positive evaluation bound required')
    plan = verify_completion(plan_path)
    cases = plan['cases']['real_evaluation']
    if not cases:
        raise ValueError('fullfit has no outer evaluation; use a separate diagnostic explicitly')
    out = Path(plan['output']) / 'evaluation'
    out.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    start = time.monotonic()
    rows = {'joint_identity': [], 'c628': []}
    try:
        for i, case in enumerate(cases):
            for arm in (('joint_identity', 'c628') if i % 2 == 0 else ('c628', 'joint_identity')):
                remaining = max_wall_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('evaluation wall cap')
                destination = out / (case + '.' + arm + '.json')
                cmd = [sys.executable, '-m', 'realpde.c628_evaluation', 'worker', '--plan', str(Path(plan_path).resolve()),
                       '--case', case, '--arm', arm, '--output', str(destination)]
                with destination.with_suffix('.log').open('x') as log:
                    subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=remaining)
                rows[arm].append(read(destination))
        scores = {arm: aggregate(r) for arm, r in rows.items()}
        write(out / 'report.json', {'status': 'COMPLETE', 'scores': scores,
              'score_delta': {k: scores['c628'][k] - scores['joint_identity'][k]
                              for k in ('rel_l2_score', 'tke_score', 'mvpe_score', 'sps_score')},
              'case_rows': rows, 'case_summary': paired_case_summary(rows), 'elapsed_seconds': time.monotonic()-start,
              'comparison': 'Same newly trained outer-excluded point; original joint identity intervals versus all C628 heads.',
              'timing_scope': 'Fresh process per case/arm; separate cold and warm predict; load outside. Official warmup boundary unverified.',
              'selection': 'Evaluation only, no refit or automatic promotion.', 'Private_or_Public_claim': False})
    except BaseException as exc:
        write(out / 'failure.json', {'status': 'INCOMPLETE', 'error': repr(exc),
                                    'completed_cases_by_arm': {a: len(r) for a, r in rows.items()},
                                    'elapsed_seconds': time.monotonic()-start})
        raise


def worker(plan_path, case, arm, destination):
    # Do not import the training recipes into the inference worker.
    plan = verify_completion(plan_path)
    if case not in plan['cases']['real_evaluation']:
        raise ValueError('case not in frozen outer evaluation')
    import numpy as np
    import torch
    from .data import inspect_case, FlowWindowDataset
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    case_info = inspect_case(Path(plan['data_root']) / 'train_real' / case)
    dataset = FlowWindowDataset([case_info], stride=20)
    pairs = [dataset[i] for i in range(len(dataset))]
    inputs = torch.stack([x for x, _ in pairs]).numpy()
    target = torch.stack([y for _, y in pairs]).numpy()
    del pairs
    runtime = Path(plan['output']) / 'export/runtime'
    # Pin runtime module imports to the exported code rather than the working src tree.
    for name in list(sys.modules):
        if name == 'realpde' or name.startswith('realpde.'):
            del sys.modules[name]
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('submission', runtime / 'submission.py')
    model = importlib.util.module_from_spec(spec)
    started = time.monotonic()
    spec.loader.exec_module(model)
    load_seconds = time.monotonic() - started
    if arm == 'c628':
        predict = model.predict
    else:
        original = model._load_head(Path(plan['output']) / 'joint/final_head.pt')
        @torch.inference_mode()
        def predict(value, metadata=None):
            parts = {'prediction': [], 'lower': [], 'upper': []}
            for i in range(0, len(value), 32):
                x = torch.from_numpy(np.ascontiguousarray(value[i:i+32])).to(model._DEVICE)
                p = model._MODEL(x).float()
                d, scale = model.flow_diagnostics(x, p)
                shift, radius = original(d.float())
                center = p[..., :2] + shift * scale
                parts['prediction'].append(p.cpu())
                parts['lower'].append(model._three_channels(center - radius * scale).cpu())
                parts['upper'].append(model._three_channels(center + radius * scale).cpu())
            return {k: torch.cat(v).numpy() for k, v in parts.items()}
    def invoke():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        begin = time.monotonic()
        values = predict(inputs, metadata={})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return values, time.monotonic() - begin
    cold, cold_seconds = invoke()
    warm, warm_seconds = invoke()
    if set(cold) != {'prediction', 'lower', 'upper'}:
        raise ValueError('invalid result keys')
    for key, value in cold.items():
        if value.shape != inputs.shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid output ' + key)
        if np.any(value[..., 2] != 0) or not np.array_equal(value, warm[key]):
            raise ValueError('pressure/repeat mismatch ' + key)
    if not np.all(cold['lower'] <= cold['upper']):
        raise ValueError('invalid bounds')
    spec = importlib.util.spec_from_file_location('official', ROOT / 'vendor/starting_kit_v9/scoring.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    errors = {key: float(fn(cold['prediction'], target, 2).mean()) for key, fn in (
        ('rel_l2', official.rel_l2_per_sample), ('tke', official.tke_rel_l2_per_sample),
        ('mvpe', official.mvpe_rel_l2_per_sample))}
    sps, coverage = official.aggregate_sps(cold['prediction'], target, 2, cold['lower'], cold['upper'])
    write(destination, {'case': case, 'arm': arm, 'windows': len(dataset),
          'mean_errors': errors, 'sps_score': float(100*sps), 'coverage': float(coverage),
          'scored_elements': int(np.count_nonzero(target[..., :2])),
          'load_seconds': load_seconds, 'cold_seconds': cold_seconds, 'warm_seconds': warm_seconds,
          'repeat_bitexact': True, 'metadata': {}, 'pid': os.getpid(), 'batch_invariance_tested': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='mode', required=True)
    e = s.add_parser('evaluate')
    e.add_argument('--plan', type=Path, required=True)
    e.add_argument('--max-wall-seconds', type=float, required=True)
    w = s.add_parser('worker')
    w.add_argument('--plan', type=Path, required=True)
    w.add_argument('--case', required=True)
    w.add_argument('--arm', choices=['joint_identity', 'c628'], required=True)
    w.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'evaluate':
        evaluate(a.plan, a.max_wall_seconds)
    else:
        worker(a.plan, a.case, a.arm, a.output)


if __name__ == '__main__':
    main()
