"""Run a historical stage 13-15 trainer unchanged (file sha256 asserted), with in-memory rebinding only.

The trainers (runs/TTAKD/train_simat.py, train_simati_focus*.py) take --run and derive ROOT from their own location
(parents[2]); nothing is hard-coded to a run directory. The file is executed as __main__ with sys.argv/sys.path set
exactly as `python SCRIPT --run DIR [--cpu-preflight]` would (same mechanism as rebind_8_12.py sd10 mode).
Holdout folds: the trainer imports its simulation module by name after putting runs/SIMALL on sys.path
(train_simat.py:78-79 `from sim_replay_all import ...`, train_simati_focus.py:83-84 `from sim_replay_focus import ...`).
With --sim-exclude-json that module (sha256 asserted) is loaded first under its own name and its module-global
discover_cases is rebound to drop the holdout's sim_excluded cases (the stage-1-7 fold rule, as rebind_8_12.py does
for sim_replay_all); the trainer's import then resolves to that module. The focus samplers derive their focus pool
from the (rebound) case list, so held regimes never enter either pool.
Every change is written to <stage>/rebind_record.json (training runs) or printed (checks).
Checks: --check import (load + rebind), --check sim-cases (print the simulation case list the rebound module yields);
--cpu-preflight runs the trainer's own one-window CPU graph/state preflight (no optimizer update).
"""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebind_8_12 import ROOT, _verified  # noqa: E402  (same pin check as stages 8-13)


def rebind_sim_module(module_path, expected, excluded):
    """Load a simulation-replay module (verified bytes) under its own name and drop `excluded` simulation cases."""
    path, data = _verified(module_path, expected)
    for entry in (ROOT / 'src', ROOT / 'scripts'):          # the trainers put these on sys.path too (train_simat.py:7)
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    name = path.stem                                       # sim_replay_all | sim_replay_focus | sim_replay_focus2
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    original, drop = module.discover_cases, frozenset(excluded)

    def discover_cases(root, *, exclude_files=()):          # same signature as realpde.data.discover_cases
        return [c for c in original(root, exclude_files=exclude_files) if c.path.name not in drop]

    module.discover_cases = discover_cases
    return module, {'module': name, 'file': str(path.relative_to(ROOT)), 'sha256': expected, 'rebound_global': 'discover_cases',
                    'change': 'drop the holdout sim_excluded cases (stage-1-7 fold rule) from the simulation list',
                    'dropped_cases': sorted(drop)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--script', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--stage-dir', required=True)
    parser.add_argument('--sim-module', help='the trainer\'s simulation module (only with --sim-exclude-json)')
    parser.add_argument('--sim-module-sha256')
    parser.add_argument('--sim-exclude-json', help='JSON list of simulation case names to drop (holdout fold)')
    parser.add_argument('--check', choices=['none', 'import', 'sim-cases'], default='none')
    parser.add_argument('--cpu-preflight', action='store_true', help="run the trainer's own --cpu-preflight")
    args = parser.parse_args(argv)
    stage = Path(args.stage_dir)
    stage = (stage if stage.is_absolute() else ROOT / stage).resolve()
    if not stage.is_dir() or not (stage / 'plan.json').is_file():
        raise SystemExit('stage directory must exist and hold the generated plan.json')
    path, data = _verified(args.script, args.sha256)
    record = {'mode': 'exact_file_as_main', 'script': str(path.relative_to(ROOT)), 'sha256': args.sha256,
              'argv': ['--run', str(stage.relative_to(ROOT))] + (['--cpu-preflight'] if args.cpu_preflight else []),
              'in_memory_rebinding': [], 'file_on_disk_unchanged': True}
    if args.sim_exclude_json:
        excluded = json.loads(Path(args.sim_exclude_json).read_text())
        module, change = rebind_sim_module(args.sim_module, args.sim_module_sha256, excluded)
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
    if not args.cpu_preflight:
        (stage / 'rebind_record.json').write_text(json.dumps(record, indent=2) + '\n')
    sys.argv = [str(path)] + record['argv']
    driver_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(path.parent)] + [p for p in sys.path if p != driver_dir]   # `python SCRIPT` semantics
    namespace = {'__name__': '__main__', '__file__': str(path), '__builtins__': __builtins__}
    exec(compile(data, str(path), 'exec'), namespace)


if __name__ == '__main__':
    main()
