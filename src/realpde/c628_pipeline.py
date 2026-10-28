"""Release-only C628 training graph with an explicit outer regime holdout.

Preparation is CPU-only. Execution requires an explicit wall-time/output bound.
The fixed historical recipes remain untouched; no old weights are training inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CASE = re.compile(r'^(\d+)_(\d+)\.h5$')
BASE_RECIPE = 'scripts/run_c252_release_only_c180_scratch_recipe.py'
RUNTIME = 'submission/c628_runtime'
COMPONENTS = {
    'model.pt': 'joint/final_point.pt',
    'center_head.pt': 'sps/sps_center/center_live.pt',
    'radius_head.pt': 'sps/radius_capacity/capacity_live.pt',
    'hidden_head.pt': 'hidden/aligned_live_hidden.pt',
    'observed_center_head.pt': 'observed/aligned_observed_live.pt',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def regime(name):
    m = CASE.fullmatch(name)
    if m is None:
        raise ValueError('invalid case name: ' + name)
    return tuple(map(int, m.groups()))


def split_inventory(sim, real, holdout_re_ge=None, holdout_aoa=None):
    if len(sim) != len(set(sim)) or len(real) != len(set(real)):
        raise ValueError('duplicate case names')
    if '7575_0.h5' in real:
        raise ValueError('excluded real duplicate present')
    def held(name):
        reynolds, angle = regime(name)
        return ((holdout_re_ge is not None and reynolds >= holdout_re_ge)
                or (holdout_aoa is not None and angle == holdout_aoa))
    parts = {'sim_train': sorted(n for n in sim if not held(n)),
             'sim_excluded': sorted(n for n in sim if held(n)),
             'real_train': sorted(n for n in real if not held(n)),
             'real_evaluation': sorted(n for n in real if held(n))}
    if not parts['sim_train'] or not parts['real_train']:
        raise ValueError('empty training set')
    if (holdout_re_ge is not None or holdout_aoa is not None) and not parts['real_evaluation']:
        raise ValueError('requested holdout has no real cases')
    re_values = sorted({regime(n)[0] for n in parts['real_train']})
    if len(re_values) < 5:
        raise ValueError('need at least five training Reynolds groups for inner OOF')
    # Match the fixed recipe's contiguous five-Reynolds-fold convention.
    q, rem = divmod(len(re_values), 5)
    folds, offset = [], 0
    for i in range(5):
        size = q + (i < rem)
        group = re_values[offset:offset + size]
        offset += size
        folds.append({'fold': i, 'held_reynolds': group,
                      'fit_cases': [n for n in parts['real_train'] if regime(n)[0] not in group],
                      'cache_cases': [n for n in parts['real_train'] if regime(n)[0] in group]})
    parts['inner_folds'] = folds
    return parts


def graph():
    nodes = [{'id': 'sim', 'parents': [], 'target_role': 'sim_train'}]
    nodes += [{'id': f'fold{i}', 'parents': ['sim'], 'target_role': 'inner_real_fit_and_cache'} for i in range(5)]
    nodes += [{'id': 'interval', 'parents': [f'fold{i}' for i in range(5)], 'target_role': 'real_train_OOF_targets'},
              {'id': 'joint', 'parents': ['sim', 'interval'], 'target_role': 'real_train'},
              {'id': 'sps', 'parents': ['joint'], 'target_role': 'real_train'},
              {'id': 'hidden', 'parents': ['joint', 'sps'], 'target_role': 'real_train'},
              {'id': 'observed', 'parents': ['joint', 'sps', 'hidden'], 'target_role': 'real_train'},
              {'id': 'export', 'parents': ['joint', 'sps', 'hidden', 'observed'], 'target_role': 'none'}]
    return nodes


def effective_config(data_root, parts):
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/track1_final_retrain_c180.yaml').read_text())
    cfg['fixed_decision'] = {'center_multiplier': 1., 'width_multiplier': 1., 'group_width_kind': 'identity',
                             'group_width_multipliers': [[1., 1.] for _ in range(20)]}
    cfg['data'].update(sim_dir=str(data_root / 'train_sim'), real_dir=str(data_root / 'train_real'),
                       expected_sim_cases=len(parts['sim_train']), expected_real_cases=len(parts['real_train']))
    cfg['provenance']['historical_development_public_used_to_freeze_interval_constants'] = False
    return cfg


def validate(plan):
    parts = plan['cases']
    expected = split_inventory(parts['sim_train'] + parts['sim_excluded'],
                               parts['real_train'] + parts['real_evaluation'],
                               **plan['holdout'])
    if parts != expected:
        raise ValueError('case partition or inner fold was altered')
    if plan['nodes'] != graph():
        raise ValueError('training dependency graph was altered')
    if set(parts['real_evaluation']) & (set(parts['sim_train']) | set(parts['real_train'])):
        raise ValueError('outer target regime reaches a training stage')
    if any(p.endswith(('.pt', '.npz', '.npy', '.zip')) for p in plan['source_hashes']):
        raise ValueError('pre-existing learned artifact is not a code input')
    if plan['components'] != COMPONENTS:
        raise ValueError('component source not in current training graph')
    if plan['config'] != effective_config(Path(plan['data_root']), parts):
        raise ValueError('fixed config or data source was altered')
    return True


def prepare(data_root, output, holdout_re_ge=None, holdout_aoa=None):
    output = Path(output).resolve()
    data_root = Path(data_root).resolve(strict=True)
    sim = [p.name for p in (data_root / 'train_sim').glob('*.h5')]
    real = [p.name for p in (data_root / 'train_real').glob('*.h5') if p.name != '7575_0.h5']
    if len(sim) != 100 or len(real) != 81:
        raise ValueError('expected released sim100 / eligible real81 before outer split')
    parts = split_inventory(sim, real, holdout_re_ge, holdout_aoa)
    cfg = effective_config(data_root, parts)
    source_paths = [ROOT / BASE_RECIPE, ROOT / 'configs/track1_final_retrain_c180.yaml',
                    ROOT / 'vendor/starting_kit_v9/scoring.py',
                    *sorted((ROOT / 'src/realpde').glob('*.py')),
                    *sorted((ROOT / RUNTIME).rglob('*'))]
    pins = {str(p.relative_to(ROOT)): digest(p) for p in source_paths if p.is_file()}
    if not (ROOT / RUNTIME / 'submission.py').exists():
        raise ValueError('frozen C628 code-only runtime missing')
    plan = {'schema': 1, 'method': 'C628_RELEASE_ONLY_FIXED_RECIPE', 'status': 'PREPARED_NOT_TRAINED',
            'data_root': str(data_root), 'output': str(output),
            'holdout': {'holdout_re_ge': holdout_re_ge, 'holdout_aoa': holdout_aoa},
            'cases': parts, 'nodes': graph(), 'components': COMPONENTS, 'source_hashes': pins,
            'config': cfg, 'historical_case_exposure': 'All released cases previously researched; prospective protocol, not fresh data.',
            'inner_sim_scope': 'Shared sim pretraining excludes OUTER regimes; inner real OOF is conditional sim-to-real adaptation.',
            'selection': 'Fixed schedules/endpoints; outer evaluation never selects weights or hyperparameters.',
            'performance_status': 'NOT_MEASURED', 'old_budget_extension': False}
    validate(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'train_cases.json', parts['real_train'])
    write(output / 'plan.json', plan)
    return plan


def check_sources(plan):
    for name, expected in plan['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('source changed since plan: ' + name)
    if read(Path(plan['output']) / 'train_cases.json') != plan['cases']['real_train']:
        raise ValueError('training case file was altered')


def execute(plan_path, max_wall_seconds, max_output_gib):
    if max_wall_seconds <= 0 or max_output_gib <= 0:
        raise ValueError('explicit positive execution bounds are required')
    lock = Path(plan_path).resolve().parent / '.execution.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    try:
        return _execute(plan_path, max_wall_seconds, max_output_gib)
    finally:
        lock.unlink()


def _execute(plan_path, max_wall_seconds, max_output_gib):
    if max_wall_seconds <= 0 or max_output_gib <= 0:
        raise ValueError('explicit positive execution bounds are required')
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    validate(plan)
    check_sources(plan)
    out = Path(plan['output'])
    if out != plan_path.parent:
        raise ValueError('plan must live in its bound output directory')
    # A completed stage can resume only with the same plan and exact output hashes.
    pin = digest(plan_path)
    previous = [read(p) for p in out.glob('attempt_*.json')]
    if any('elapsed_seconds' not in r or r.get('status') == 'RUNNING' for r in previous):
        raise RuntimeError('unclosed prior attempt; account for it before resuming')
    spent = sum(float(r['elapsed_seconds']) for r in previous)
    available = max_wall_seconds - spent
    if available <= 0:
        raise TimeoutError('cumulative pipeline budget exhausted')
    start = time.monotonic()
    attempt = out / ('attempt_' + str(time.time_ns()) + '.json')
    record = {'pid': os.getpid(), 'plan_sha256': pin, 'max_wall_seconds': max_wall_seconds,
              'max_output_gib': max_output_gib, 'prior_attempt_seconds': spent, 'status': 'RUNNING', 'nodes': []}
    write(attempt, record)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        for node in plan['nodes']:
            stage = node['id']
            seal = out / stage / 'complete.json'
            if seal.exists():
                saved = read(seal)
                if saved['plan_sha256'] != pin or any(digest(out / p) != h for p, h in saved['outputs'].items()):
                    raise ValueError('completed stage provenance changed: ' + stage)
                record['nodes'].append({'stage': stage, 'status': 'REUSED_VERIFIED'})
                continue
            if (out / stage).exists():
                raise FileExistsError('partial stage retained; inspect it and prepare an explicit recovery: ' + stage)
            remaining = available - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('pipeline wall budget exhausted')
            command = [sys.executable, '-m', 'realpde.c628_pipeline', 'worker', '--plan', str(plan_path),
                       '--stage', stage, '--wall-seconds', str(remaining)]
            with (out / (stage + '.log')).open('ab') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record['current_stage'] = stage
                record['child_pid'] = child.pid
                write(attempt, record)
                try:
                    while child.poll() is None:
                        elapsed = time.monotonic() - start
                        if elapsed >= available:
                            raise TimeoutError('pipeline wall budget exhausted')
                        allocated = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                        if allocated > max_output_gib * 1024**3:
                            raise RuntimeError('pipeline output cap exceeded')
                        # Child recipes enforce device/cache limits; include worker host RSS here.
                        status = Path(f'/proc/{child.pid}/status')
                        if status.exists():
                            rss = re.search(r'^VmRSS:\s+(\d+) kB', status.read_text(), re.M)
                            if rss and int(rss.group(1)) * 1024 > 20 * 1024**3:
                                raise MemoryError('worker exceeded20GiB host RSS')
                        time.sleep(.25)
                    if child.returncode:
                        raise RuntimeError(f'{stage} worker failed: {child.returncode}; see {stage}.log')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
            if not seal.exists():
                raise RuntimeError('worker did not seal its outputs: ' + stage)
            record['nodes'].append({'stage': stage, 'status': 'COMPLETE'})
            write(attempt, record)
        record['status'] = 'TRAINING_AND_EXPORT_COMPLETE_EVALUATION_REQUIRED'
    except BaseException as exc:
        record['status'] = 'FAILED_OR_RESOURCE_STOPPED'
        record['exception'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['cumulative_seconds'] = spent + record['elapsed_seconds']
        record.pop('child_pid', None)
        write(attempt, record)


def export(plan, destination):
    out = Path(plan['output'])
    stage = Path(destination)
    stage.mkdir(parents=True, exist_ok=False)
    for p in (ROOT / RUNTIME).rglob('*'):
        if p.is_file():
            target = stage / p.relative_to(ROOT / RUNTIME)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    for name, source in COMPONENTS.items():
        shutil.copyfile(out / source, stage / name)
    size = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
    if size >= 256_000_000:
        raise ValueError('unpacked runtime exceeds256MB')
    archive = stage.parent / 'C628_RETRAINED_UNVERIFIED.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(stage)))
    return archive


def worker(plan_path, stage, wall_seconds):
    import signal
    if wall_seconds <= 0:
        raise ValueError('positive worker wall bound required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('worker wall cap')))
    signal.setitimer(signal.ITIMER_REAL, wall_seconds)
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    validate(plan)
    check_sources(plan)
    out = Path(plan['output'])
    destination = out / stage
    if out != plan_path.parent:
        raise ValueError('worker plan/output directory mismatch')
    if stage not in {n['id'] for n in plan['nodes']}:
        raise ValueError('unknown stage')
    node = next(n for n in plan['nodes'] if n['id'] == stage)
    pin = digest(plan_path)
    for parent in node['parents']:
        completed = read(out / parent / 'complete.json')
        if completed['plan_sha256'] != pin or any(digest(out / p) != h for p, h in completed['outputs'].items()):
            raise ValueError('invalid parent stage: ' + parent)
    if stage == 'export':
        destination.mkdir(exist_ok=False)
        archive = export(plan, destination / 'runtime')
        write(destination / 'report.json', {'status': 'UNVERIFIED', 'archive': str(archive),
                                             'sha256': digest(archive), 'fresh_model_inference': False})
    elif stage in {'sps', 'hidden', 'observed'}:
        shared = ['--point', str(out / 'joint/final_point.pt'), '--real-dir', plan['config']['data']['real_dir'],
                  '--cases', str(out / 'train_cases.json'), '--output', str(destination),
                  '--gpu-seconds', str(wall_seconds), '--wall-seconds', str(wall_seconds)]
        if stage == 'sps':
            from .sps_interval_recipe import main
            shared += ['--joint-head', str(out / 'joint/final_head.pt')]
        else:
            shared += ['--center', str(out / COMPONENTS['center_head.pt']), '--radius', str(out / COMPONENTS['radius_head.pt'])]
            if stage == 'hidden':
                from .fixed_hidden_sps_training import main
            else:
                from .fixed_observed_center_training import main
                shared += ['--hidden', str(out / COMPONENTS['hidden_head.pt'])]
        main(shared)
    else:
        base_worker(plan, stage, destination)
    outputs = {str(p.relative_to(out)): digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write(destination / 'complete.json', {'stage': stage, 'plan_sha256': pin, 'outputs': outputs,
                                          'outer_targets_used_for_training': False})


def base_worker(plan, stage, destination):
    """Reuse fixed learning functions; replace only full81-specific orchestration."""
    import torch
    from scripts import run_c252_release_only_c180_scratch_recipe as base
    from .fast_interval import FastIntervalConfig, FastIntervalHead
    cfg = plan['config']
    out = Path(plan['output'])
    if stage != 'sim':
        from .c628_ram_data import install_real_read_cache
        install_real_read_cache(base)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = base.choose_device(cfg.get('device', 'auto'))
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 8 * 1024**3 / total), device)
    base.seed_everything(int(cfg['seed']))
    def cases(domain, names):
        available = {c.path.name: c for c in base.discover_cases(Path(plan['data_root']) / domain,
                                                               exclude_files=['7575_0.h5'] if domain == 'train_real' else [])}
        if not set(names) <= available.keys():
            raise ValueError('training case missing')
        return [available[n] for n in names]
    real = cases('train_real', plan['cases']['real_train'])
    data = cfg['data']
    def save_head(path, model, metadata):
        torch.save({'model_config': model.config.to_dict(), 'state_dict': model.state_dict(), 'metadata': metadata}, path)
    if stage == 'sim' or stage.startswith('fold'):
        is_sim = stage == 'sim'
        settings = cfg['training']['sim' if is_sim else 'oof_point']
        if is_sim:
            train_cases = cases('train_sim', plan['cases']['sim_train'])
            seed = int(settings['seed'])
            base.seed_everything(seed)
            model = base.HybridFlowOperator(base.ModelConfig.from_dict(cfg['model'], data)).to(device)
        else:
            fold = int(stage[4:])
            entry = plan['cases']['inner_folds'][fold]
            train_cases = cases('train_real', entry['fit_cases'])
            seed = int(cfg['seed']) + int(settings['seed_offset']) + fold
            model = base.load_point(out / 'sim/point.pt', device)
        dataset = base.flow_dataset(train_cases, cfg, stride=int(data['train_stride']))
        model, report = base.train_point(model, dataset, cfg, device, steps=int(settings['steps']),
            batch_size=int(settings['batch_size']), learning_rate=float(settings['learning_rate']),
            minimum_learning_rate=float(settings['minimum_learning_rate']), seed=seed, stage=stage)
        base.save_point(destination / 'point.pt', model, report)
        if not is_sim:
            held_cases = cases('train_real', entry['cache_cases'])
            cache_data = base.flow_dataset(held_cases, cfg, stride=int(data['oof_validation_stride']))
            _, report['cache'] = base.cache_fold(model, cache_data, real, device, cfg, fold=fold,
                held_reynolds=entry['held_reynolds'], path=destination / 'cache.pt', batch_limit=None)
        if not report['all_finite']:
            raise FloatingPointError('nonfinite point checkpoint')
    elif stage == 'interval':
        cache = base.load_oof([out / f'fold{i}/cache.pt' for i in range(5)])
        head, report = base.train_head(cache, cfg, device, steps=int(cfg['training']['interval_head']['steps']))
        if not report['all_finite']:
            raise FloatingPointError('nonfinite interval checkpoint')
        save_head(destination / 'head.pt', head, report)
    elif stage == 'joint':
        ck = torch.load(out / 'interval/head.pt', map_location='cpu', weights_only=False)
        head = FastIntervalHead(FastIntervalConfig.from_dict(ck['model_config'])).to(device)
        head.load_state_dict(ck['state_dict'], strict=True)
        model, head, report, dataset = base.train_joint(out / 'sim/point.pt', head, real, cfg, device,
                                                       steps=int(cfg['training']['joint_fullfit']['steps']))
        if not report['point_all_finite'] or not report['head_all_finite']:
            raise FloatingPointError('nonfinite joint checkpoint')
        base.save_point(destination / 'final_point.pt', model, report)
        save_head(destination / 'final_head.pt', head, report)
        report['bounds'] = base.bounds_smoke(model, head, dataset, cfg, device)
        if not report['bounds']['all_finite'] or not report['bounds']['bounds_ordered']:
            raise ValueError('invalid joint bounds')
    else:
        raise ValueError('unknown base stage: ' + stage)
    write(destination / 'report.json', {'status': 'COMPLETE', 'training': report,
                                        'outer_evaluation_cases': plan['cases']['real_evaluation'],
                                        'outer_targets_used': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--holdout-re-ge', type=int)
    prep.add_argument('--holdout-aoa', type=int)
    run = sub.add_parser('execute')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--max-wall-seconds', type=float, required=True)
    run.add_argument('--max-output-gib', type=float, required=True)
    node = sub.add_parser('worker')
    node.add_argument('--plan', type=Path, required=True)
    node.add_argument('--stage', required=True)
    node.add_argument('--wall-seconds', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        p = prepare(args.data_root, args.output, args.holdout_re_ge, args.holdout_aoa)
        print(json.dumps({'status': p['status'], 'cases': {k: len(v) for k, v in p['cases'].items()}, 'nodes': len(p['nodes'])}))
    elif args.command == 'execute':
        execute(args.plan, args.max_wall_seconds, args.max_output_gib)
    else:
        worker(args.plan, args.stage, args.wall_seconds)


if __name__ == '__main__':
    main()
