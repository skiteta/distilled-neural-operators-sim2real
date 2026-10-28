"""Explicit Re/AoA list holdout for the C675 weight-free pipeline, applied in memory.

The stock `python -m realpde.c628_pipeline` supports only `--holdout-re-ge X` (threshold)
and a single `--holdout-aoa A`; the driver uses the stock CLI whenever the requested
holdout yields the identical sim+real partition. For other lists this module:
  * generalizes `split_inventory` (same checks, same contiguous five-Reynolds inner folds);
  * mirrors stock `prepare` and records the lists in plan['holdout'] (the stock validate
    then refuses such a plan with a TypeError, so it cannot be run without this module);
  * runs the STOCK validate/check_sources/execute/worker/export with that generalization
    installed, routing each worker subprocess through this file so it is installed there too.
`src/realpde/c628_pipeline.py` is never edited (existing C675 plans pin it).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
for entry in (str(ROOT), str(ROOT / 'src')):
    if entry not in sys.path:
        sys.path.insert(0, entry)
from realpde import c628_pipeline as P  # noqa: E402

STOCK_SPLIT = P.split_inventory
STOCK_WORKER = [sys.executable, '-m', 'realpde.c628_pipeline', 'worker']


def split_inventory(sim, real, holdout_re_ge=None, holdout_aoa=None, holdout_re_values=None, holdout_aoa_values=None):
    if not holdout_re_values and not holdout_aoa_values:
        return STOCK_SPLIT(sim, real, holdout_re_ge, holdout_aoa)
    if len(sim) != len(set(sim)) or len(real) != len(set(real)):
        raise ValueError('duplicate case names')
    if '7575_0.h5' in real:
        raise ValueError('excluded real duplicate present')
    re_values = {int(v) for v in holdout_re_values or ()}
    aoa_values = {int(v) for v in holdout_aoa_values or ()} | ({int(holdout_aoa)} if holdout_aoa is not None else set())

    def held(name):
        reynolds, angle = P.regime(name)
        return (holdout_re_ge is not None and reynolds >= holdout_re_ge) or reynolds in re_values or angle in aoa_values

    kept = STOCK_SPLIT([n for n in sim if not held(n)], [n for n in real if not held(n)])  # stock folds + checks
    parts = {'sim_train': kept['sim_train'], 'sim_excluded': sorted(n for n in sim if held(n)),
             'real_train': kept['real_train'], 'real_evaluation': sorted(n for n in real if held(n)),
             'inner_folds': kept['inner_folds']}
    if not parts['real_evaluation']:
        raise ValueError('requested holdout has no real cases')
    return parts


class _SubprocessShim:
    """Replaces c628_pipeline's `subprocess` global: only the worker command is rerouted."""
    TimeoutExpired = subprocess.TimeoutExpired

    @staticmethod
    def Popen(command, *args, **kwargs):
        if list(command[:4]) != STOCK_WORKER:
            raise RuntimeError('unexpected subprocess from c628_pipeline: ' + ' '.join(map(str, command[:4])))
        return subprocess.Popen([sys.executable, str(HERE), 'worker', *command[4:]], *args, **kwargs)


def install():
    P.split_inventory = split_inventory
    P.subprocess = _SubprocessShim


def prepare(data_root, output, holdout):
    """Mirror of stock c628_pipeline.prepare with list-valued holdout keys."""
    install()
    holdout = {'holdout_re_ge': holdout.get('holdout_re_ge'), 'holdout_aoa': holdout.get('holdout_aoa'),
               'holdout_re_values': sorted(int(v) for v in holdout.get('holdout_re_values') or []),
               'holdout_aoa_values': sorted(int(v) for v in holdout.get('holdout_aoa_values') or [])}
    output = Path(output).resolve()
    data_root = Path(data_root).resolve(strict=True)
    sim = [p.name for p in (data_root / 'train_sim').glob('*.h5')]
    real = [p.name for p in (data_root / 'train_real').glob('*.h5') if p.name != '7575_0.h5']
    if len(sim) != 100 or len(real) != 81:
        raise ValueError('expected released sim100 / eligible real81 before outer split')
    parts = split_inventory(sim, real, **holdout)
    cfg = P.effective_config(data_root, parts)
    source_paths = [ROOT / P.BASE_RECIPE, ROOT / 'configs/track1_final_retrain_c180.yaml',
                    ROOT / 'vendor/starting_kit_v9/scoring.py', *sorted((ROOT / 'src/realpde').glob('*.py')),
                    *sorted((ROOT / P.RUNTIME).rglob('*')), HERE]
    pins = {str(p.relative_to(ROOT)): P.digest(p) for p in source_paths if p.is_file()}
    if not (ROOT / P.RUNTIME / 'submission.py').exists():
        raise ValueError('frozen C628 code-only runtime missing')
    plan = {'schema': 1, 'method': 'C628_RELEASE_ONLY_FIXED_RECIPE', 'status': 'PREPARED_NOT_TRAINED',
            'data_root': str(data_root), 'output': str(output), 'holdout': holdout,
            'cases': parts, 'nodes': P.graph(), 'components': P.COMPONENTS, 'source_hashes': pins,
            'config': cfg, 'historical_case_exposure': 'All released cases previously researched; prospective protocol, not fresh data.',
            'inner_sim_scope': 'Shared sim pretraining excludes OUTER regimes; inner real OOF is conditional sim-to-real adaptation.',
            'selection': 'Fixed schedules/endpoints; outer evaluation never selects weights or hyperparameters.',
            'performance_status': 'NOT_MEASURED', 'old_budget_extension': False,
            'holdout_extension': str(HERE.relative_to(ROOT))}
    P.validate(plan)
    output.mkdir(parents=True, exist_ok=False)
    P.write(output / 'train_cases.json', parts['real_train'])
    P.write(output / 'plan.json', plan)
    return plan


def check(plan_path):
    install()
    plan = P.read(plan_path)
    P.validate(plan)
    P.check_sources(plan)
    if Path(plan['output']) != Path(plan_path).resolve().parent:
        raise ValueError('plan must live in its bound output directory')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--holdout-json', required=True)
    run = sub.add_parser('execute')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--max-wall-seconds', type=float, required=True)
    run.add_argument('--max-output-gib', type=float, required=True)
    node = sub.add_parser('worker')
    node.add_argument('--plan', type=Path, required=True)
    node.add_argument('--stage', required=True)
    node.add_argument('--wall-seconds', type=float, required=True)
    chk = sub.add_parser('check')
    chk.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    install()
    if args.command == 'prepare':
        plan = prepare(args.data_root, args.output, json.loads(args.holdout_json))
        print(json.dumps({'status': plan['status'], 'cases': {k: len(v) for k, v in plan['cases'].items()},
                          'nodes': len(plan['nodes'])}))
    elif args.command == 'execute':
        P.execute(args.plan, args.max_wall_seconds, args.max_output_gib)
    elif args.command == 'worker':
        P.worker(args.plan, args.stage, args.wall_seconds)
    else:
        plan = check(args.plan)
        print(json.dumps({'status': 'PLAN_VALID_SOURCES_UNCHANGED', 'holdout': plan['holdout'],
                          'cases': {k: len(v) for k, v in plan['cases'].items()}}))


if __name__ == '__main__':
    main()
