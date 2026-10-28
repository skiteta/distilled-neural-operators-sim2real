"""Run a historical stage-6/7 train.py unchanged, with its module-global OUT rebound.

train.py derives OUT from its own location and main() reads the globals ROOT/OUT at call
time, so importing the file as a module, setting OUT to a fresh stage directory under ROOT
and calling main(seconds, arm) runs the historical code without editing it (spec S6 a).

Full release: the file is executed exactly as on disk (sha256 asserted first).
Regime holdout (fold) of C758R train.py only: that file hard-codes 81 training cases
(train.py:46,47,55,98). With --train-case-count N the wrapper applies an in-memory,
count-only substitution (each target must occur exactly once), injects
EXPECTED_TRAIN_CASES=N and records the transformation; the file on disk is never changed.
C787R train.py already derives the count from the plan and needs no substitution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C758_FOLD_SUBSTITUTIONS = [
    ("assert len(m['train_cases'])==81 and", "assert len(m['train_cases'])==EXPECTED_TRAIN_CASES and"),
    ("and len(cases)==81 and", "and len(cases)==EXPECTED_TRAIN_CASES and"),
    ("'all_cases':81,", "'all_cases':EXPECTED_TRAIN_CASES,"),
    ("'train_cases':81,", "'train_cases':EXPECTED_TRAIN_CASES,"),
]
C758_SHA256 = '1f5de6c0f4f28243af568473a4153d009fe4f6ef8754439bc18e5c1588e17847'


def load_script(script, expected_sha256, train_case_count=None):
    script = Path(script).resolve(strict=True)
    data = script.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise SystemExit(f'code pin mismatch for {script}: {actual} != {expected_sha256}')
    name = 'rebuild_' + script.parent.name.split('_')[0].lower() + '_train'
    record = {'script': str(script.relative_to(ROOT)), 'sha256': actual}
    if train_case_count is None:
        spec = importlib.util.spec_from_file_location(name, script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record['mode'] = 'exact_file'
        return module, record
    if actual != C758_SHA256:
        raise SystemExit('the fold count substitution is defined only for C758R train.py')
    if train_case_count == 81 or train_case_count < 5:
        raise SystemExit('fold substitution needs a holdout case count (use the exact file for 81)')
    text = data.decode('utf-8')
    for old, new in C758_FOLD_SUBSTITUTIONS:
        if text.count(old) != 1:
            raise SystemExit(f'substitution target not found exactly once: {old!r}')
        text = text.replace(old, new)
    module = types.ModuleType(name)
    module.__file__ = str(script)               # ROOT/OUT derive from the real location
    module.EXPECTED_TRAIN_CASES = int(train_case_count)
    sys.modules[name] = module
    exec(compile(text, str(script), 'exec'), module.__dict__)
    record.update(mode='in_memory_count_substitution', expected_train_cases=int(train_case_count),
                  substitutions=[{'old': o, 'new': n} for o, n in C758_FOLD_SUBSTITUTIONS],
                  transformed_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    return module, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--arm', default='candidate', choices=['candidate', 'previous'])
    parser.add_argument('--seconds', type=float, default=0.0)
    parser.add_argument('--train-case-count', type=int)
    parser.add_argument('--check', choices=['none', 'import', 'plan'], default='none',
                        help='import: load+rebind only; plan: also the train.py:25 source-pin check (no GPU)')
    args = parser.parse_args(argv)
    script = args.script if args.script.is_absolute() else ROOT / args.script
    module, record = load_script(script, args.sha256, args.train_case_count)
    stage = args.stage_dir.resolve()
    if Path(module.ROOT).resolve() != ROOT:
        raise SystemExit('script ROOT differs from the driver ROOT')
    stage.relative_to(ROOT)                     # train.py:95,97 need OUT under ROOT
    if not stage.is_dir():
        raise SystemExit('stage directory must exist and hold the generated plan')
    module.OUT = stage
    record.update(stage_dir=str(stage.relative_to(ROOT)), arm=args.arm)
    if args.check == 'import':
        print(json.dumps({'status': 'IMPORT_REBIND_OK', **record}))
        return
    plan_path = module.OUT / (args.arm + '_plan.json')
    plan = module.read(plan_path)
    bad = [p for p, h in plan['source_pins'].items() if module.sha(module.ROOT / p) != h]
    if bad:
        raise SystemExit('source pins changed: ' + ', '.join(bad[:5]))
    if (module.OUT / ('training_' + args.arm)).exists():
        raise SystemExit('training directory already exists (train.py refuses it)')
    if args.check == 'plan':
        print(json.dumps({'status': 'PLAN_CHECK_OK', 'source_pins': len(plan['source_pins']), **record}))
        return
    (stage / 'rebind_record.json').write_text(json.dumps(record, indent=2) + '\n')
    module.main(args.seconds, args.arm)


if __name__ == '__main__':
    main()
