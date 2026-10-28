"""Run a historical stage 8-13 training script unchanged (file sha256 asserted), with in-memory rebinding only.

Modes (the executed bytes are the verified bytes; the files on disk are never changed):
  latcadf  runs/LATCADF_full81/train.py derives OUT from its own location (train.py:4) and reads
           OUT/'candidate_plan.json' (train.py:40). As rebind_train.py does for C758R/C787R, the file is executed
           as a module, the module-global OUT is rebound to the stage directory and main(seconds, 'candidate') is
           called. The script directory is put first on sys.path exactly as `python train.py` would, because
           train.py:33 imports reflection_transform from there (pinned in the plan's source_pins).
  sd10     runs/SD10/train_sd10.py, runs/CYCLE4/train_cycle4.py, runs/SIMALL/train_sim.py take --run and
           derive ROOT from their location (parents[2]); nothing is hard-coded to a run directory. The file is
           executed as __main__ with sys.argv/sys.path set as `python SCRIPT --run DIR [--cpu-preflight]` would.
           Holdout folds + sim replay: sim_replay_all.py (policy 'all' = every released simulation,
           sim_replay_all.py:19) is loaded first (sha256 asserted) and its module-global discover_cases is
           rebound to drop the holdout's sim_excluded cases, i.e. the stage-1-7 fold rule (held regimes never
           enter training). train_sim.py:76 `from sim_replay_all import ...` then resolves to that module.
Every change is written to <stage>/rebind_record.json (training runs) or printed (checks).
Checks: --check import (load + rebind), --check plan (+ source pins + no existing training dir),
        --check sim-cases (sd10: print the simulation case list the rebound module yields), and for the sd10
        family --cpu-preflight (the scripts' own one-window CPU graph/state preflight, no optimizer update).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _verified(script, expected):
    script = Path(script)
    script = (script if script.is_absolute() else ROOT / script).resolve(strict=True)
    data = script.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise SystemExit(f'code pin mismatch for {script}: {actual} != {expected}')
    return script, data


def rebind_sim_replay(module_path, expected, excluded):
    """Load sim_replay_all.py (verified bytes) as the module 'sim_replay_all' and drop `excluded` sim cases."""
    path, data = _verified(module_path, expected)
    for entry in (ROOT / 'src', ROOT / 'scripts'):          # train_sim.py:5 / :76 put these on sys.path too
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    module = types.ModuleType('sim_replay_all')
    module.__file__ = str(path)
    sys.modules['sim_replay_all'] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    original, drop = module.discover_cases, frozenset(excluded)

    def discover_cases(root, *, exclude_files=()):          # same signature as realpde.data.discover_cases
        return [c for c in original(root, exclude_files=exclude_files) if c.path.name not in drop]

    module.discover_cases = discover_cases
    return module, {'module': 'sim_replay_all', 'file': str(path.relative_to(ROOT)), 'sha256': expected,
                    'rebound_global': 'discover_cases',
                    'change': 'drop the holdout sim_excluded cases (stage-1-7 fold rule) from the policy-all simulation list',
                    'dropped_cases': sorted(drop)}


def load_latcadf(script, expected, stage):
    path, data = _verified(script, expected)
    stage = Path(stage).resolve()
    stage.relative_to(ROOT)
    sys.path[0] = str(path.parent)          # `python train.py` semantics: reflection_transform (train.py:33)
    name = 'rebuild_latcadf_train'
    module = types.ModuleType(name)
    module.__file__ = str(path)             # ROOT/OUT derive from the real location first (train.py:4)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    if Path(module.ROOT).resolve() != ROOT:
        raise SystemExit('script ROOT differs from the driver ROOT')
    original_out = str(Path(module.OUT).relative_to(ROOT))
    module.OUT = stage
    record = {'mode': 'latcadf_module_out_rebinding', 'script': str(path.relative_to(ROOT)), 'sha256': expected,
              'rebound_globals': {'OUT': {'from': original_out, 'to': str(stage.relative_to(ROOT))}},
              'sys_path_0': str(path.parent.relative_to(ROOT)), 'file_on_disk_unchanged': True}
    return module, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['latcadf', 'sd10'])
    parser.add_argument('--script', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--stage-dir', required=True)
    parser.add_argument('--seconds', type=float, default=0.0, help='latcadf: TrainingBudget seconds (caps unused by train.py)')
    parser.add_argument('--sim-module', help='sd10: runs/SIMALL/sim_replay_all.py (only with --sim-exclude-json)')
    parser.add_argument('--sim-module-sha256')
    parser.add_argument('--sim-exclude-json', help='sd10: JSON list of simulation case names to drop (holdout fold)')
    parser.add_argument('--check', choices=['none', 'import', 'plan', 'sim-cases'], default='none')
    parser.add_argument('--cpu-preflight', action='store_true', help='sd10: run the script with --cpu-preflight')
    args = parser.parse_args(argv)
    stage = Path(args.stage_dir)
    stage = (stage if stage.is_absolute() else ROOT / stage).resolve()
    if not stage.is_dir():
        raise SystemExit('stage directory must exist and hold the generated plan')
    if args.mode == 'latcadf':
        module, record = load_latcadf(args.script, args.sha256, stage)
        if args.check == 'import':
            print(json.dumps({'status': 'IMPORT_REBIND_OK', **record}))
            return
        plan = module.read(module.OUT / 'candidate_plan.json')
        bad = [p for p, h in plan['source_pins'].items() if module.sha(module.ROOT / p) != h]
        if bad:
            raise SystemExit('source pins changed: ' + ', '.join(bad[:5]))
        if (module.OUT / 'training_candidate').exists():
            raise SystemExit('training directory already exists (train.py:40 refuses it)')
        if args.check == 'plan':
            print(json.dumps({'status': 'PLAN_CHECK_OK', 'source_pins': len(plan['source_pins']), **record}))
            return
        (stage / 'rebind_record.json').write_text(json.dumps(record, indent=2) + '\n')
        module.main(args.seconds, 'candidate')
        return
    # sd10 family: exact file executed as __main__
    path, data = _verified(args.script, args.sha256)
    record = {'mode': 'exact_file_as_main', 'script': str(path.relative_to(ROOT)), 'sha256': args.sha256,
              'argv': ['--run', str(stage.relative_to(ROOT))] + (['--cpu-preflight'] if args.cpu_preflight else []),
              'in_memory_rebinding': [], 'file_on_disk_unchanged': True}
    if args.sim_exclude_json:
        excluded = json.loads(Path(args.sim_exclude_json).read_text())
        module, change = rebind_sim_replay(args.sim_module, args.sim_module_sha256, excluded)
        record['in_memory_rebinding'].append(change)
        if args.check == 'sim-cases':
            cases = module.discover_cases(ROOT / 'data/extracted/train_sim', exclude_files=['7575_0.h5'])
            print(json.dumps({'status': 'SIM_CASES', 'cases': [c.path.name for c in cases], **record}))
            return
    elif args.check == 'sim-cases':
        raise SystemExit('--check sim-cases needs --sim-exclude-json')
    if args.check == 'import':
        print(json.dumps({'status': 'IMPORT_OK', **record}))
        return
    if args.check == 'plan':
        raise SystemExit('--check plan is latcadf-only; use --cpu-preflight for the sd10 family')
    if not args.cpu_preflight:
        (stage / 'rebind_record.json').write_text(json.dumps(record, indent=2) + '\n')
    sys.argv = [str(path)] + record['argv']
    sys.path[0] = str(path.parent)          # `python SCRIPT` semantics
    namespace = {'__name__': '__main__', '__file__': str(path), '__builtins__': __builtins__}
    exec(compile(data, str(path), 'exec'), namespace)


if __name__ == '__main__':
    main()
