"""FINAL_REBUILD driver, stages 8-12 (+ optional 13): stage-7 output (C787H-equivalent runtime) -> LC4BH-equivalent package.

README
======
Continues a COMPLETE `rebuild_1_7.py` output directory and reproduces the rest of the LC4BH training lineage from
scratch. Every stage consumes only the previous stage's rebuilt, sealed outputs. Code is pinned by sha256 and asserted
(common_8_12.CODE_PINS_8_12 = the static import closure of the four training scripts + the SD10 reflection source;
runtime_code_c973h/ = the vendored non-weight runtime members). Historical weights / checkpoints / ZIPs are never read
by a rebuild; their hashes and stream digests are recorded for comparison (common_8_12.HISTORICAL_8_12) and --smoke
reads the historical run directories as read-only stand-ins.

Usage (any cwd; relative paths are relative to the project root; interpreter .venv310/bin/python)
  rebuild  rebuild_8_12.py --stage7 runs/FINAL_REBUILD/<R1> --out runs/FINAL_REBUILD/<R1>_8_12
           [--holdout-re LIST] [--holdout-aoa LIST] [--seed N] [--extra-cycle] [--sim-replay all]
           [--stop-before-gpu] [--resume]
  smoke    rebuild_8_12.py --smoke --out runs/FINAL_REBUILD/smoke_8_12/<name> [--stage7 DIR] [same options]
           [--keep-smoke-outputs]
  --stage7         COMPLETE rebuild_1_7.py output: rebuild_report.json COMPLETE/rebuild, seals s6_c758h + s7_c787h
                   verified, s7_c787h/package_audit.json archive sha checked. Optional with --smoke (default stand-in:
                   the historical C787H package, read-only).
  --holdout-re/--holdout-aoa  same syntax and partition code as rebuild_1_7.py (c675_ext). With --stage7 the fold is
                   inherited from it (given flags must match). real_train = training cases of every stage and of
                   every KD stream; real_evaluation = plan eval_cases (names only; the scripts assert disjointness and
                   never load them); sim_excluded (simulations at held regimes) is dropped from the simulation replay
                   by in-memory rebinding of sim_replay_all.discover_cases (rebind_8_12.py). As in rebuild_1_7, h5
                   metadata (frame counts) of all release files is read by discover_cases.
  --seed N         recipe seed of stages 8-13 (default 20260919799, the full81 lineage). The historical held surfaces
                   used 20260919796 (upperRe, --holdout-re '>=19050'), ...797 (AoA15), ...798 (AoA20).
  --extra-cycle    stage 13 LC5BH: LC5-style continuation (train_cycle4.py, KD from stage 12, stream offset 200 =
                   runs/CYCLE5/rule.json LC5B; LC5B itself only ran on held surfaces).
  --sim-replay all the FINAL continuation cycle is trained by runs/SIMALL/train_sim.py with training.sim_replay='all'
                   (SIMA variant, simulation relL2 weight 0.1). With --extra-cycle: stage 13 = SIMAH, the historical
                   lineage LC4BH -> SIMAH (runs/SIMA_full81/plan.json). Without it: stage 12 = LC4BSIMH (LC4B cycle,
                   offset 100, + simulation replay), a combination that was never trained historically.
  --stop-before-gpu  rebuild mode: generate plans and run all CPU checks, stop before the first GPU training launch.
  --resume         continue in --out; sealed stages are verified (sha256) and reused, regenerated plans must equal
                   the stored ones. A partial training_candidate/ (the scripts refuse an existing one) or a partial
                   runtime/ must be moved aside by hand.

Stages (historical training code runs unchanged; plans are generated here; packaging is lifted: package_8_12.py)
  s8  LATCADFH  runs/LATCADF_full81/train.py via `rebind_8_12.py latcadf` (module OUT rebound to the stage dir), init =
                stage-7 runtime, no teacher; lattice 2x2 phases, cadence-2 p=0.25, dropout 0.03, composite loss;
                steps 2*S6 (18464), schedule 4*S6, midpoint S6, EMA .999**(6000/S6), S6 = ceil(6000*windows/42847).
  s9  LCSDH     runs/SD10/train_sd10.py (exact file as __main__), KD from s8, same data/reflection streams as s8.
  s10 LCSD2H    runs/SD10/train_sd10.py, KD from s9.
  s11 LCSD3H    runs/SD10/train_sd10.py, KD from s10 (teacher package = the previous rebuilt package; no override).
  s12 LC4BH     runs/CYCLE4/train_cycle4.py, KD from s11, stream_seed_offset 100, replay_teacher_stream false.
  s13 LC5BH | SIMAH (optional, see above).
  Per stage: plan -> CPU checks (source pins; s8: `--check plan` + runtime states/12 Dropout(.03)/2 hidden.detach;
  s9+: the script's own --cpu-preflight, one window) -> CPU replay of every seeded stream -> GPU training -> the
  report's stream digests must equal the CPU replay -> package_8_12 -> seal. Stage dir: <out>/sN_<candidate>/ with
  plan.json (+ candidate_plan.json for s8), training_script.txt, training_candidate/, runtime/, <CAND>_REBUILT.zip,
  deployed checkpoint, package_audit.json. Final deliverable: <out>/s12_lc4bh/runtime and LC4BH_REBUILT.zip (or s13).
  Blockers removed: B1 plans pin only code + the previous rebuilt runtime; B2 payload templates and carried
  training_recipe/ come from the previous rebuilt package, runtime code from runtime_code_c973h/; B3 no byte-identical
  handoff to historical files (all handoffs chain-internal and sealed); B5 <CAND>_train.py.txt = the script that trained
  the stage (training_script.txt); B6 generated plans, no teacher_path_override, no copied plans, no absolute paths.
  Not reproduced: finish_generic.py/finalize_*.py evaluation, qualification and timing (not training), and the F5
  runtime variants (runs/EXACTLANE/make_fixed5.py edits submission.py and adds fixed_transfer.py after packaging).

GPU time, one RTX A5500 (common_8_12.GPU_ESTIMATES_8_12, from training_seconds in the historical curve.jsonl/train.log):
  s8 ~1.6k s unshared (9,287 s logged, shared); s9-s12 ~1.9k s each (LCSD3H ran alone: 1,893 s; LCSDH 5,834,
  LCSD2H 5,609, LC4BH 11,065 s shared); s8-s12 ~9.2k s (2.6 h) unshared, 33.7k s (9.4 h) as logged; SIMAH +2.7k s
  (11,269 s logged); LC5BH +1.9k s. Packaging: seconds on CPU. Folds scale with 2*S6 steps.

Verification (--smoke: CPU only; GPU training dry-run with its exact command). Results: <out>/rebuild_report.json and
<out>/verification/*.json (kept; bulky smoke files are removed unless --keep-smoke-outputs).
  * plans generated from stand-in previous-stage outputs (historical, read-only) are compared on every plan path the
    training script, sim_replay_all.py and the historical packager read (AST scan): each difference must be an
    intended one (source_pins, runtime/teacher paths, teacher_archive, runtime_baseline) or the smoke fails; the full
    training and sampling dicts must be equal.
  * every seeded stream (source windows, reflection, lattice phase, cadence incl. trailing fallbacks, simulation
    windows) is replayed on CPU and compared with the historical resume_*.pt / report.json digests and counts.
  * packaging: package_8_12 runs on each stage's historical training outputs and is compared member by member with
    the historical package ZIP (s12: submissions/20260924_131335_JST__READY__LC4BH_LATCAD_SD4.zip); every member that is
    not byte-identical needs a listed reason (PACKAGE_REASONS) or the smoke fails.
  * folds: partition as rebuild_1_7; upperRe/AoA15/AoA20 folds are compared with the historical held-surface plans
    and streams when --seed is the held seed.
  Verified 2026-09-24 (runs/FINAL_REBUILD/smoke_8_12/*): full release s8-s13 plans 0 unexpected differences; all
  stream digests/counts equal to the historical runs (LATCADFH..LC4BH, SIMAH incl. simulation stream); packaging of the
  historical LC4B outputs reproduces 31/33 members of the READY LC4BH ZIP byte-identically (32/33 with the historical
  plan embedded) and the deployed checkpoint byte-identically; held upperRe/AoA15/AoA20 folds (with their held seeds)
  match all historical held plans (read fields) and streams; rebuild mode was exercised on CPU up to the first GPU
  launch, incl. a mocked s8 -> s9 handoff, resume/reuse and seal-tamper detection. Suite: smoke_8_12/run_suite.sh.
Untested (no GPU used): the GPU training steps themselves; packaging of freshly trained (non-historical) outputs;
  bitwise weight reproducibility across runs/GPUs (stream equality is expected, weight equality is not); a real
  rebuild_1_7.py output as --stage7 (the stage-1-7 GPU run had not completed); organizer hardware (allocator caps:
  4 GiB per process in these scripts, SIMA peaked at 3.9 GiB).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, DRIVER, ENV, EXCLUDED_REAL, REBUILD_BASE, RETAINED_REAL, ROOT, compare, read_json,  # noqa: E402
                    regime, rel, release_names, sha256, write_json)
from common_8_12 import (CODE_PINS_8_12, GPU_ESTIMATES_8_12, HELD_FOLD_SEEDS, HIST, HIST_C758_PLAN, HIST_C787H_ZIP,  # noqa: E402
                         HIST_HELD, HISTORICAL_8_12, LIFTED_FROM, RUNTIME_C973, RUNTIME_C973_DIR,
                         RUNTIME_C973_ORDER, RUNTIME_C973_SOURCE, SAMPLING_META, SD10_REFLECTION, SEED, SIM_MODULE,
                         SIM_VARIANT, STAGES, T8, T_SIM, TRAIN_SCRIPTS, WEIGHTS, estimate_seconds, latcadf_training,
                         plan_fields_read, script_closure, sd10_training)
import c675_ext  # noqa: E402  (same partition code as rebuild_1_7; imports realpde.c628_pipeline, no torch)
from rebuild_1_7 import parse_aoa, parse_re  # noqa: E402

DRIVER_FILES = ['common.py', 'common_8_12.py', 'rebuild_8_12.py', 'rebind_8_12.py', 'preflight_8_12.py',
                'package_8_12.py', 'preflight.py', 'c675_ext.py', 'rebuild_1_7.py']
PACKAGER_FIELDS = {'latcadf': ('runs/LATCAD_upperRe/package_latcad.py', ('m',)),
                   'sd10': ('runs/SD10/finalize_sd10.py', ('m', 'teacher', 'fast'))}
ALIASES = {'teacher': ('m', 'teacher_archive'), 'fast': ('m', 'runtime_baseline'), 'manifest': ('m',)}
INTENDED = {
    ('m', 'source_pins'): 'B1: code closure + previous rebuilt runtime; historical pins held non-lineage weights (C1017R/C1031R)',
    ('m', 'runtime'): 'path of the previous stage runtime of this rebuild (smoke: stand-in extraction)',
    ('m', 'teacher_source'): 'previous stage directory of this rebuild',
    ('m', 'teacher_archive'): 'previous stage package of this rebuild (payload templates, carried training_recipe)',
    ('m', 'runtime_baseline'): 'B2: vendored runtime_code_c973h/ instead of the weight-bearing C973H ZIP',
}
MISSING = '<absent>'
PACKAGE_REASONS = {   # every package member that is not byte-identical to the historical ZIP needs one of these
    'continuation': 'intended: <CAND>_continuation.json embeds the rebuilt plan (rebuild paths, code+runtime source_pins, '
                    'vendored runtime_baseline) instead of the historical plan',
    'b5': 'intended (B5): <CAND>_train.py.txt is the script that trained LC4BH, runs/CYCLE4/train_cycle4.py; the historical '
          'ZIP carries runs/SD10/train_sd10.py under this name',
    'sim_module': 'explained: runs/SIMALL/sim_replay_all.py was edited 2026-09-24 13:34 UTC, after SIMAH was packaged (13:14): '
                  'the loss weight reads training.sim_weight with default .1 (unset in the SIMA plan) instead of the constant .1',
    'rename': 'explained: the historical LCSDH ZIP used the pre-fix recipe names training_recipe/sd10_*; finalize_sd10.py '
              'switched to <CAND>_* after the LCSD2H duplicate-member failure (renamed members are byte-identical)',
    'b2_payload': 'intended (B2): weight payload template = the stage-7 (C787H) package instead of the non-lineage C1017H ZIP; '
                  'state_dict identical, payload lacks the C1017H provenance keys reflection_fraction/reflection_point_lr',
    'b2_payload_rebuilt_s7': 'intended (B2) with a rebuilt stage 7: as b2_payload, and the stage-7 provenance records '
                             '(schedule_continuation / C728 / C758 / metadata) carry the rebuilt values; model_config identical',
}
PROVENANCE_KEYS = {'schedule_continuation', 'C728', 'C758', 'metadata'}


class StageError(RuntimeError):
    pass


class StopBeforeGpu(Exception):
    """--stop-before-gpu: everything up to the first GPU training launch is done (plans, CPU checks)."""


def dig(value, keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def compute_holdout(re_values, re_ge, aoa_values):
    """rebuild_1_7.Driver.resolve_holdout, without the writes (same c675_ext calls, same result keys)."""
    sim, real = release_names('train_sim'), release_names('train_real')
    all_re = {regime(n)[0] for n in sim + real}
    requested = {'holdout_re_values': re_values, 'holdout_re_ge': re_ge, 'holdout_aoa_values': aoa_values}
    native, ext = {}, None
    if not re_values and re_ge is None and not aoa_values:
        mode, parts = 'none', c675_ext.STOCK_SPLIT(sim, real)
    else:
        parts = c675_ext.split_inventory(sim, real, holdout_re_ge=re_ge, holdout_re_values=re_values,
                                         holdout_aoa_values=aoa_values)
        native = None
        if len(aoa_values) <= 1:
            held_re = sorted(set(re_values) | {r for r in all_re if re_ge is not None and r >= re_ge})
            candidate = {'holdout_re_ge': held_re[0] if held_re else None, 'holdout_aoa': aoa_values[0] if aoa_values else None}
            try:
                native = candidate if c675_ext.STOCK_SPLIT(sim, real, **candidate) == parts else None
            except ValueError:
                native = None
        mode = 'c675_native' if native is not None else 'c675_list_extension'
        if native is None:
            ext = {'holdout_re_ge': re_ge, 'holdout_aoa': None, 'holdout_re_values': re_values, 'holdout_aoa_values': aoa_values}
    return {'mode': mode, 'requested': requested, 'c675_native_args': native, 'c675_list_holdout': ext,
            'counts': {k: len(v) for k, v in parts.items()},
            'cases': {k: parts[k] for k in ('sim_train', 'sim_excluded', 'real_train', 'real_evaluation')}}


class Driver:
    def __init__(self, args):
        out = Path(args.out)
        out = (out if out.is_absolute() else ROOT / out).resolve()
        if REBUILD_BASE not in out.parents or out == DRIVER or DRIVER in out.parents:
            raise SystemExit('--out must be a new directory under runs/FINAL_REBUILD/ (not driver/)')
        if args.smoke and args.resume:
            raise SystemExit('--smoke cannot be combined with --resume')
        if out.exists() != bool(args.resume):
            raise SystemExit('--out must not exist (or must exist with --resume): ' + str(out))
        if not args.smoke and not args.stage7:
            raise SystemExit('--stage7 (a COMPLETE rebuild_1_7.py output) is required unless --smoke')
        self.stage7_dir = None
        if args.stage7:
            s7 = Path(args.stage7)
            self.stage7_dir = (s7 if s7.is_absolute() else ROOT / s7).resolve(strict=True)
        self.out, self.args, self.smoke = out, args, args.smoke
        self.owned = False
        self.py = args.python or sys.executable
        self.flags_given = bool(args.holdout_re or args.holdout_aoa)
        self.re_values, self.re_ge = parse_re(args.holdout_re)
        self.aoa_values = parse_aoa(args.holdout_aoa)
        names = release_names('train_sim') + release_names('train_real')           # as rebuild_1_7.py
        unknown = ([v for v in self.re_values if v not in {regime(n)[0] for n in names}]
                   + [v for v in self.aoa_values if v not in {regime(n)[1] for n in names}])
        if unknown:
            raise SystemExit(f'holdout values not in the release: {unknown}')
        self.seed, self.sim, self.extra = args.seed, args.sim_replay == 'all', args.extra_cycle
        self.keys = ['s8', 's9', 's10', 's11', 's12'] + (['s13'] if self.extra else [])
        self.history = False                  # smoke: historical lineage present (full release, recipe seed)
        self.report = {'status': 'RUNNING', 'mode': 'smoke_cpu' if self.smoke else 'rebuild', 'out': None,
                       'options': {'seed': self.seed, 'sim_replay': args.sim_replay, 'extra_cycle': self.extra},
                       'stages': {}, 'plan_comparison': {}, 'stream_comparison': {}, 'package_parity': {},
                       'historical_comparison': {}, 'gpu_steps': [], 'commands': []}
        self.started = time.monotonic()

    # ---------------------------------------------------------------- utilities (rebuild_1_7 conventions)
    def log(self, message):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
        print(line, flush=True)
        if self.owned:
            with (self.out / 'driver.log').open('a') as f:
                f.write(line + '\n')

    def env(self, cpu=False):
        env = dict(os.environ)
        env.update(ENV)
        env.pop('PYTHONPATH', None)
        if cpu or self.smoke:
            env['CUDA_VISIBLE_DEVICES'] = ''
        return env

    def run(self, name, argv, cpu=False):
        argv = [str(a) for a in argv]
        log = self.out / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        self.log(f'RUN {name}' + (' [CPU]' if cpu or self.smoke else ' [GPU]'))
        with log.open('ab') as stream:
            stream.write(('\n$ ' + ' '.join(argv) + '\n').encode())
            stream.flush()
            proc = subprocess.run(argv, cwd=ROOT, env=self.env(cpu), stdout=stream, stderr=subprocess.STDOUT)
        seconds = round(time.monotonic() - start, 1)
        self.report['commands'].append({'name': name, 'argv': argv, 'returncode': proc.returncode, 'seconds': seconds,
                                        'log': rel(log), 'cuda_hidden': bool(cpu or self.smoke)})
        if proc.returncode:
            raise StageError(f'{name} failed (exit {proc.returncode}); see {rel(log)}')
        self.log(f'OK  {name} ({seconds} s)')
        return log

    @staticmethod
    def last_json(log):
        for line in reversed(log.read_text(errors='replace').splitlines()):
            if line.startswith('{'):
                return json.loads(line)
        return None

    def stage(self, name, **values):
        self.report['stages'].setdefault(name, {}).update(values)

    def gpu_step(self, key, kind, steps, argv):
        est = estimate_seconds(kind, steps, key)
        self.report['gpu_steps'].append({'stage': key, 'kind': kind, 'steps': steps, 'command': [str(a) for a in argv],
                                         'estimate_seconds': {'unshared': est[0], 'historical_logged_shared': est[1]},
                                         'note': GPU_ESTIMATES_8_12[kind]['note']})
        self.log(f'DRY-RUN {key}: GPU step not run (one A5500: ~{est[0]} s unshared; historical wall {est[1]} s shared)')

    def gpu_launch(self, key, argv):
        if self.args.stop_before_gpu:
            self.stage(key, status='STOPPED_BEFORE_GPU', gpu_command=[str(a) for a in argv])
            raise StopBeforeGpu(key)
        self.run(f'{key}_train', argv)

    def seal(self, key, paths):
        write_json(self.out / 'seals' / (key + '.json'), {'stage': key, 'outputs': {rel(p): sha256(p) for p in paths}})

    def verify_seal(self, path):
        bad = [p for p, h in read_json(path)['outputs'].items() if sha256(ROOT / p) != h]
        if bad:
            raise StageError(f'sealed outputs changed ({rel(path)}): {bad[:3]}')

    def sealed(self, key):
        path = self.out / 'seals' / (key + '.json')
        if not path.exists():
            return False
        self.verify_seal(path)
        self.log(f'REUSE {key} (seal verified)')
        self.stage(key, status='REUSED_SEALED')
        return True

    def verification(self, name, value):
        write_json(self.out / 'verification' / (name + '.json'), value)

    def full_release(self):
        return self.holdout['mode'] == 'none'

    # ---------------------------------------------------------------- stage table
    def spec(self, key):
        s = dict(STAGES[key], key=key, sim=False)
        if self.sim and key == self.keys[-1]:
            s.update(SIM_VARIANT[key], script=T_SIM, sim=True)
        s['dir'] = f"{key}_{s['candidate'].lower()}"
        # historical full81 counterpart (HIST key): s8-s12 as such; SIMAH for s13+sim; none for LC5BH / LC4BSIMH
        if key == 's13':
            s['hist'] = 's13_sim' if s['sim'] else None
        else:
            s['hist'] = None if s['sim'] else key
        return s

    # ---------------------------------------------------------------- prologue
    def prologue(self):
        manifest = {'stage7': rel(self.stage7_dir) if self.stage7_dir else None, 'holdout_re': self.re_values,
                    'holdout_re_ge': self.re_ge, 'holdout_aoa': self.aoa_values, 'seed': self.seed,
                    'sim_replay': self.args.sim_replay, 'extra_cycle': self.extra, 'smoke': self.smoke, 'python': self.py}
        if self.args.resume:
            if not (self.out / 'driver_manifest.json').exists():
                raise SystemExit('--resume needs an existing driver_manifest.json in --out')
            if {k: v for k, v in read_json(self.out / 'driver_manifest.json').items() if k != 'argv'} != manifest:
                raise SystemExit('--resume arguments differ from the original run')
            self.owned = True
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            self.owned = True
            write_json(self.out / 'driver_manifest.json', {'argv': sys.argv, **manifest})
        self.report['out'] = rel(self.out)
        self.log(f'START {self.report["mode"]} out={rel(self.out)} stages={self.keys} sim_replay={self.args.sim_replay} seed={self.seed}')
        for path, expected in CODE_PINS_8_12.items():                              # code pins (asserted)
            if sha256(ROOT / path) != expected:
                raise StageError('code pin mismatch: ' + path)
        for script in TRAIN_SCRIPTS:                                               # closure unchanged (asserted)
            extra = set(script_closure(script)) - set(CODE_PINS_8_12)
            if extra:
                raise StageError(f'import closure of {script} gained unpinned files: {sorted(extra)}')
        vendored = {str(p.relative_to(RUNTIME_C973_DIR)): sha256(p) for p in sorted(RUNTIME_C973_DIR.rglob('*')) if p.is_file()}
        if vendored != RUNTIME_C973:
            raise StageError('vendored runtime_code_c973h differs from its pins')
        names = release_names('train_real')                                        # data layout (asserted)
        assert len(names) == 81 and RETAINED_REAL in names and EXCLUDED_REAL not in names
        assert len(release_names('train_sim')) == 100  # Decision-phase data omits train_real/7575_0.h5 (always excluded)
        if not (self.out / 'environment.json').exists():
            probe = ("import json,platform,sys,torch,numpy,h5py;print(json.dumps({'python':sys.version.split()[0],"
                     "'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),"
                     "'numpy':numpy.__version__,'h5py':h5py.__version__,'platform':platform.platform()}))")
            log = self.run('environment_probe', [self.py, '-c', probe], cpu=True)
            files = sorted(CODE_PINS_8_12) + [str((DRIVER / f).relative_to(ROOT)) for f in DRIVER_FILES]
            write_json(self.out / 'environment.json', {
                'software': self.last_json(log), 'env': ENV, 'code_sha256': {f: sha256(ROOT / f) for f in files},
                'vendored_runtime_code': {'dir': rel(RUNTIME_C973_DIR), 'sha256': RUNTIME_C973, 'source': RUNTIME_C973_SOURCE},
                'lifted_packaging_code_recorded': LIFTED_FROM})

    # ---------------------------------------------------------------- stage-7 input and holdout
    def resolve_inputs(self):
        if self.stage7_dir is not None:
            s7 = self.load_stage7()
        else:
            holdout = compute_holdout(self.re_values, self.re_ge, self.aoa_values)
            s7 = {'dir': None, 'stand_in': True, 'holdout': holdout}
        self.holdout = s7['holdout']
        cases = self.holdout['cases']
        self.train_cases, self.eval_cases = cases['real_train'], cases['real_evaluation']
        self.sim_excluded = cases['sim_excluded']
        write_json(self.out / 'holdout.json', self.holdout)
        write_json(self.out / 'inputs' / 'train_cases.json', self.train_cases)
        write_json(self.out / 'inputs' / 'sim_excluded.json', self.sim_excluded)
        self.run('inventory', [self.py, DRIVER / 'preflight_8_12.py', 'inventory', '--cases-json',
                               self.out / 'inputs' / 'train_cases.json', '--sim-excluded-json',
                               self.out / 'inputs' / 'sim_excluded.json', '--out', self.out / 'inputs' / 'inventory.json'], cpu=True)
        self.inv = read_json(self.out / 'inputs' / 'inventory.json')
        self.base, self.windows = self.inv['base_steps'], self.inv['windows']
        if s7['dir'] is not None and (s7['base'], s7['windows']) != (self.base, self.windows):
            raise StageError('stage-7 S6 plan steps/windows differ from the inventory rule')
        self.fold = next((t for t, kw in (('upperRe', {'holdout_re_ge': 19050}), ('AoA15', {'holdout_aoa': 15}),
                                          ('AoA20', {'holdout_aoa': 20}))
                          if self.holdout['mode'] != 'none' and c675_ext.STOCK_SPLIT(
                              release_names('train_sim'), release_names('train_real'), **kw)['real_train'] == self.train_cases), None)
        if s7['stand_in']:
            s7.update(self.standin_stage7())
        self.s7 = s7
        self.history = self.smoke and self.full_release() and self.seed == SEED and all(
            (ROOT / h[k]).exists() for h in HIST.values() for k in ('dir', 'plan', 'zip'))
        self.log(f'HOLDOUT mode={self.holdout["mode"]} counts={self.holdout["counts"]} windows={self.windows} '
                 f'base(S6)={self.base} held_surface={self.fold} stage7={"stand-in" if s7["stand_in"] else rel(s7["dir"])}')
        self.stage('inputs', holdout_mode=self.holdout['mode'], counts=self.holdout['counts'], windows=self.windows,
                   base_steps=self.base, stage_steps=2 * self.base, sim_cases=len(self.inv['sim']),
                   sim_windows=self.inv['sim_windows'], stage7=None if s7['stand_in'] else rel(s7['dir']),
                   stage7_runtime=rel(s7['runtime']), stage7_archive=rel(s7['archive']), stand_in=s7['stand_in'],
                   held_surface_match=self.fold, history_available=self.history)

    def load_stage7(self):
        d = self.stage7_dir
        report = read_json(d / 'rebuild_report.json')
        if report.get('status') != 'COMPLETE' or report.get('mode') != 'rebuild':
            raise StageError(f'{rel(d)} is not a COMPLETE rebuild_1_7.py output')
        for key in ('s6_c758h', 's7_c787h'):
            self.verify_seal(d / 'seals' / (key + '.json'))
        m7 = read_json(d / 'driver_manifest.json')
        if self.flags_given and (sorted(self.re_values), self.re_ge, sorted(self.aoa_values)) != (
                sorted(m7['holdout_re']), m7['holdout_re_ge'], sorted(m7['holdout_aoa'])):
            raise StageError('--holdout flags differ from the stage-7 run (the fold is inherited from it)')
        self.re_values, self.re_ge, self.aoa_values = m7['holdout_re'], m7['holdout_re_ge'], m7['holdout_aoa']
        holdout7 = read_json(d / 'holdout.json')
        mine = compute_holdout(self.re_values, self.re_ge, self.aoa_values)
        if any(holdout7[k] != mine[k] for k in ('mode', 'counts', 'cases')):
            raise StageError('stage-7 holdout.json differs from the recomputed partition')
        p6 = read_json(d / 's6_c758h' / 'candidate_plan.json')
        audit = read_json(d / 's7_c787h' / 'package_audit.json')
        runtime, archive = ROOT / audit['runtime'], ROOT / audit['archive']
        if sha256(archive) != audit['archive_sha256']:
            raise StageError('stage-7 archive differs from its sealed package_audit.json')
        if p6['train_cases'] != mine['cases']['real_train']:
            raise StageError('stage-6 plan train_cases differ from the holdout partition')
        return {'dir': d, 'stand_in': False, 'holdout': mine, 'runtime': runtime, 'archive': archive,
                'archive_sha256': audit['archive_sha256'], 'base': p6['training']['steps'], 'windows': p6['sampling']['windows']}

    def standin_stage7(self):
        """Smoke only: the historical C787H package (read-only) extracted as the stage-7 runtime stand-in."""
        if not (ROOT / HIST_C787H_ZIP).exists():
            raise StageError(f'--smoke without --stage7 needs the historical {HIST_C787H_ZIP} as a read-only stand-in')
        runtime = self.extract('s7', ROOT / HIST_C787H_ZIP)
        record = {'runtime': runtime, 'archive': ROOT / HIST_C787H_ZIP, 'archive_sha256': sha256(ROOT / HIST_C787H_ZIP)}
        if self.full_release() and (ROOT / HIST_C758_PLAN).exists():
            hist = read_json(ROOT / HIST_C758_PLAN)
            self.report['historical_comparison']['s6_steps_rule'] = compare(self.base, hist['training']['steps'])
            self.report['historical_comparison']['s6_windows'] = compare(self.windows, hist['sampling']['windows'])
        return record

    def extract(self, key, archive):
        target = self.out / 'standins' / key / 'runtime'
        if not target.exists():
            self.run(f'standin_extract_{key}', [self.py, DRIVER / 'preflight_8_12.py', 'extract', '--zip', archive,
                                               '--out', target, '--record', target.parent / 'extract.json'], cpu=True)
        return target

    # ---------------------------------------------------------------- plans
    def pins(self, script, runtime, extra=()):
        files = sorted(set(script_closure(script)) | set(extra))
        pins = {f: CODE_PINS_8_12[f] for f in files}
        members = sorted(p for p in Path(runtime).rglob('*') if p.is_file())
        if any('__pycache__' in p.parts for p in members):
            raise StageError(f'{rel(runtime)} contains __pycache__ (a runtime must hold package members only)')
        pins.update({rel(p): sha256(p) for p in members})
        return pins

    def common_fields(self):
        sampling = {'candidate': SAMPLING_META['candidate'], 'stride': SAMPLING_META['stride'],
                    'support_frames': SAMPLING_META['support_frames'], 'windows': self.windows,
                    'interpretation': SAMPLING_META['interpretation']}      # key order of the historical plans
        return {'train_cases': self.train_cases, 'eval_cases': self.eval_cases, 'sampling': sampling}

    def holdout_record(self):
        return {'mode': self.holdout['mode'], 'requested': self.holdout['requested'], 'counts': self.holdout['counts'],
                'real_evaluation': self.eval_cases, 'sim_excluded': self.sim_excluded,
                'rule': 'real_train only for training/KD streams; eval_cases never loaded; sim_excluded never replayed'}

    def runtime_baseline(self):
        return {'code_dir': rel(RUNTIME_C973_DIR), 'members': RUNTIME_C973, 'order': RUNTIME_C973_ORDER,
                'historical_zip_recorded_only': RUNTIME_C973_SOURCE}

    def plan8(self, s7):
        base, spec = self.base, self.spec('s8')
        plan = {'candidate': spec['candidate'], 'generated_by': rel(DRIVER / 'rebuild_8_12.py'),
                'policy': 'LATCADF recipe inlined (runs/LATCADF_full81/candidate_plan.json): steps=2*S6, schedule=4*S6, '
                          'middle=S6, EMA=.999**(6000/S6), S6=ceil(6000*windows/42847); init = stage-7 runtime; no teacher',
                **self.common_fields(), 'training': latcadf_training(base, self.seed), 'middle_step': base, 'resume_step': 0,
                'runtime': rel(s7['runtime']), 'source_pins': self.pins(T8, s7['runtime']),
                'teacher_archive': {'path': rel(s7['archive']), 'sha256': s7['archive_sha256'],
                                    'role': 'packaging only: weight payload templates = the stage-7 package (no teacher in training)'},
                'runtime_baseline': self.runtime_baseline(), 'holdout': self.holdout_record(), 'stage_note': spec['note'],
                'lineage': {'previous': 'stage 7 (C787H-equivalent)', 'stage7': None if s7['stand_in'] else rel(s7['dir']),
                            'stand_in': s7['stand_in']}}
        return plan

    def plan_sd10(self, spec, prev):
        extra = [SD10_REFLECTION]
        training = sd10_training(self.base, self.seed, spec['offset'], spec['replay'], 'all' if spec['sim'] else None)
        plan = {'candidate': spec['candidate'], 'generated_by': rel(DRIVER / 'rebuild_8_12.py'),
                'policy': 'SD10/CYCLE4 recipe inlined (runs/LCSD_full81, runs/LC4B_full81, runs/SIMA_full81 plans): '
                          'finetune=2*S6 (no warm-up), middle=S6, EMA=.999**(6000/S6); teacher = previous rebuilt stage',
                **self.common_fields(), 'training': training, 'middle_step': self.base,
                'runtime': rel(prev['runtime']), 'teacher_source': rel(prev['dir']),
                'teacher_archive': {'path': rel(prev['zip']), 'sha256': prev['zip_sha256']},
                'reflection_source': SD10_REFLECTION, 'runtime_baseline': self.runtime_baseline(),
                'source_pins': self.pins(spec['script'], prev['runtime'], extra), 'holdout': self.holdout_record(),
                'stage_note': spec['note'],
                'lineage': {'previous': prev['key'], 'previous_candidate': prev['candidate'], 'stand_in': prev['stand_in']}}
        if spec['sim']:
            plan['sim_replay_scope'] = {'policy': 'all', 'excluded_by_holdout': self.sim_excluded,
                                        'cases': len(self.inv['sim']), 'windows': self.inv['sim_windows']}
        return plan

    def write_plan(self, d, name, plan):
        path = d / name
        if path.exists():
            if read_json(path) != plan:
                raise StageError(f'{rel(path)} differs from the regenerated plan; inspect and move the stage aside')
        else:
            write_json(path, plan)
        return path

    # ---------------------------------------------------------------- CPU checks shared by smoke and rebuild
    def replay(self, key, d, plan_name, kind):
        target = d / 'preflight_streams.json'
        if not target.exists():
            self.run(f'{key}_replay', [self.py, DRIVER / 'preflight_8_12.py', 'replay', '--plan', d / plan_name,
                                       '--inventory', self.out / 'inputs' / 'inventory.json', '--kind', kind, '--out', target], cpu=True)
        return read_json(target)

    def sd10_rebind(self, spec, d, check=None, preflight=False):
        argv = [self.py, DRIVER / 'rebind_8_12.py', 'sd10', '--script', spec['script'],
                '--sha256', CODE_PINS_8_12[spec['script']], '--stage-dir', d]
        if spec['sim'] and self.sim_excluded:
            argv += ['--sim-module', SIM_MODULE, '--sim-module-sha256', CODE_PINS_8_12[SIM_MODULE],
                     '--sim-exclude-json', self.out / 'inputs' / 'sim_excluded.json']
        if check:
            argv += ['--check', check]
        if preflight:
            argv += ['--cpu-preflight']
        return argv

    def latcadf_rebind(self, d, check=None):
        argv = [self.py, DRIVER / 'rebind_8_12.py', 'latcadf', '--script', T8, '--sha256', CODE_PINS_8_12[T8],
                '--stage-dir', d, '--seconds', '0']
        return argv + (['--check', check] if check else [])

    # ---------------------------------------------------------------- stage 8
    def stage8(self):
        spec, key = self.spec('s8'), 's8'
        d = self.out / spec['dir']
        if self.sealed(key):
            self.record_outputs(key, spec, d)
            return self.prev_from_stage(key)
        plan = self.plan8(self.s7)
        t = d / 'training_candidate'
        if d.exists() and t.exists() and not (t / 'report.json').exists():
            raise StageError(f'{rel(t)} is a partial training run; move it aside, then --resume')
        d.mkdir(exist_ok=True)
        self.write_plan(d, 'candidate_plan.json', plan)
        self.write_plan(d, 'plan.json', plan)          # teacher plan read by stage 9 (train_sd10.py:96)
        trained = (t / 'report.json').exists()
        if not trained:
            check = self.run('s8_rebind_check', self.latcadf_rebind(d, 'plan'), cpu=True)
            self.stage(key, rebind=self.last_json(check))
            if not (d / 'preflight_runtime.json').exists():
                self.run('s8_runtime_check', [self.py, DRIVER / 'preflight_8_12.py', 'runtime', '--plan', d / 'candidate_plan.json',
                                              '--out', d / 'preflight_runtime.json'], cpu=True)
        streams = self.replay(key, d, 'candidate_plan.json', 'latcadf')
        runtime = read_json(d / 'preflight_runtime.json') if (d / 'preflight_runtime.json').exists() else None
        self.stage(key, candidate=spec['candidate'], plan=rel(d / 'candidate_plan.json'), plan_sha256=sha256(d / 'candidate_plan.json'),
                   steps=plan['training']['steps'], middle_step=plan['middle_step'], EMA=plan['training']['EMA'],
                   windows=self.windows, initial_states=runtime and runtime['initial_states'],
                   predicted_streams={k: streams[k] for k in ('batch_digest', 'augmentation_digest', 'phase_digest', 'cadence_digest',
                                                              'phase_counts', 'cadence_counts', 'cadence_trailing_fallbacks')})
        self.compare_streams(key, 's8', streams, d, 'candidate_plan.json')
        if runtime and self.full_release() and self.seed == SEED:
            self.report['historical_comparison']['s8_initial_states'] = compare(runtime['initial_states'], HISTORICAL_8_12['s8']['initial_states'])
        if self.smoke:
            self.compare_plan(key, spec, plan, None)
            self.gpu_step(key, 'latcadf', plan['training']['steps'], self.latcadf_rebind(d))
            if self.history:
                self.parity(key, spec, d / 'candidate_plan.json')
            self.stage(key, status='PLAN_GENERATED_CPU_CHECKS_OK__TRAIN_DRY_RUN')
            return self.standin_prev(key, spec, d)
        if not trained:
            self.gpu_launch(key, self.latcadf_rebind(d))
        report = read_json(t / 'report.json')
        if report['status'] != 'COMPLETE' or report['steps_actual'] != plan['training']['steps']:
            raise StageError('S8 training report not COMPLETE')
        actual = {k: report[k] for k in ('phase_digest', 'cadence_digest', 'augmentation_digest', 'phase_counts',
                                         'cadence_counts', 'cadence_trailing_fallbacks')}
        mismatch = [k for k, v in actual.items() if streams[k] != v]
        if mismatch:
            raise StageError(f'S8 training consumed other streams than the CPU replay predicted: {mismatch}')
        return self.package(key, spec, d, 'latcadf', 'candidate_plan.json', report)

    # ---------------------------------------------------------------- stages 9-13
    def stage_sd10(self, key, prev):
        spec = self.spec(key)
        d = self.out / spec['dir']
        if self.sealed(key):
            self.record_outputs(key, spec, d)
            return self.prev_from_stage(key)
        if not prev['stand_in']:
            self.verify_seal(self.out / 'seals' / (prev['key'] + '.json'))
        plan = self.plan_sd10(spec, prev)
        t = d / 'training_candidate'
        if d.exists() and t.exists() and not (t / 'report.json').exists():
            raise StageError(f'{rel(t)} is a partial training run; move it aside, then --resume')
        d.mkdir(exist_ok=True)
        self.write_plan(d, 'plan.json', plan)
        script_spec = spec['script'] + ('\n' + SIM_MODULE if spec['sim'] else '') + '\n'
        if (d / 'training_script.txt').exists() and (d / 'training_script.txt').read_text() != script_spec:
            raise StageError(f'{rel(d)}/training_script.txt differs')
        (d / 'training_script.txt').write_text(script_spec)
        trained = (t / 'report.json').exists()
        if not trained:
            pre = d / 'cpu_preflight' / 'report.json'
            if (d / 'cpu_preflight').exists() and not (pre.exists() and read_json(pre)['source_plan_sha256'] == sha256(d / 'plan.json')):
                raise StageError(f'{rel(d)}/cpu_preflight is stale or partial; move it aside')
            if not pre.exists():
                self.run(f'{key}_cpu_preflight', self.sd10_rebind(spec, d, preflight=True), cpu=True)
            result = read_json(pre)
            if result['status'] != 'PASS':
                raise StageError(f'{key} CPU preflight did not pass')
            self.stage(key, cpu_preflight={k: result[k] for k in ('status', 'student_depth', 'teacher_depth', 'student_parameters', 'actual_seconds')})
            if spec['sim'] and self.sim_excluded:
                log = self.run(f'{key}_sim_rebind_check', self.sd10_rebind(spec, d, check='sim-cases'), cpu=True)
                got = self.last_json(log)['cases']
                if got != [n for n, _ in self.inv['sim']]:
                    raise StageError('rebound simulation list differs from the inventory')
                self.stage(key, sim_rebinding={'cases': len(got), 'dropped': len(self.sim_excluded)})
        streams = self.replay(key, d, 'plan.json', 'sd10')
        self.stage(key, candidate=spec['candidate'], script=spec['script'], plan=rel(d / 'plan.json'), plan_sha256=sha256(d / 'plan.json'),
                   teacher=prev['candidate'], teacher_stand_in=prev['stand_in'], steps=plan['training']['steps'],
                   stream_seed_offset=plan['training'].get('stream_seed_offset', 0),
                   replay_teacher_stream=plan['training'].get('replay_teacher_stream', True), sim_replay=plan['training'].get('sim_replay'),
                   predicted_streams={k: streams[k] for k in ('batch_digest', 'augmentation_digest', 'lattice_phase_counts',
                                                              'cadence_counts', 'cadence_trailing_fallbacks')},
                   predicted_simulation=streams.get('simulation_replay'))
        self.compare_streams(key, spec['hist'] or key, streams, d, 'plan.json')
        kind = 'sim' if spec['sim'] else 'sd10'
        if self.smoke:
            self.compare_plan(key, spec, plan, prev)
            self.gpu_step(key, kind, plan['training']['steps'], self.sd10_rebind(spec, d))
            if self.history and spec['hist']:
                self.parity(key, spec, d / 'plan.json')
            self.stage(key, status='PLAN_GENERATED_CPU_PREFLIGHT_OK__TRAIN_DRY_RUN')
            return self.standin_prev(key, spec, d) if key != self.keys[-1] else None
        if not trained:
            self.gpu_launch(key, self.sd10_rebind(spec, d))
        report = read_json(t / 'report.json')
        if report['status'] != 'COMPLETE' or report['steps_actual'] != plan['training']['steps']:
            raise StageError(f'{key} training report not COMPLETE')
        phase = report['phases'][-1]
        mismatch = [k for k in ('batch_digest', 'augmentation_digest', 'lattice_phase_counts', 'cadence_counts',
                                'cadence_trailing_fallbacks') if phase[k] != streams[k]]
        if spec['sim'] and report['simulation_replay']['digest'] != streams['simulation_replay']['digest']:
            mismatch.append('simulation_replay.digest')
        if mismatch:
            raise StageError(f'{key} training consumed other streams than the CPU replay predicted: {mismatch}')
        return self.package(key, spec, d, 'sd10', 'plan.json', report)

    def package(self, key, spec, d, kind, plan_name, report):
        audit_path = d / 'package_audit.json'
        if not audit_path.exists():
            if (d / 'runtime').exists():
                raise StageError(f'{rel(d)}/runtime exists without package_audit.json; move it aside, then --resume')
            self.run(f'{key}_package', [self.py, DRIVER / 'package_8_12.py', kind, '--plan', d / plan_name, '--out', d], cpu=True)
        audit = read_json(audit_path)
        t = d / 'training_candidate'
        plan = read_json(d / plan_name)
        paths = [d / plan_name, t / 'report.json', t / 'job.json', t / f"resume_{plan['middle_step']}.pt",
                 t / f"resume_{plan['training']['steps']}.pt", audit_path, ROOT / audit['archive'], ROOT / audit['deployed_checkpoint'],
                 *[p for p in (d / 'runtime').rglob('*') if p.is_file()]]
        paths.append(d / ('plan.json' if kind == 'latcadf' else 'training_script.txt'))
        self.seal(key, paths)
        self.record_outputs(key, spec, d)
        return self.prev_from_stage(key)

    def record_outputs(self, key, spec, d):
        """Stage record + historical comparison (recorded, never asserted); also used when a sealed stage is reused."""
        audit = read_json(d / 'package_audit.json')
        report = read_json(d / 'training_candidate' / 'report.json')
        self.stage(key, runtime=audit['runtime'], archive=audit['archive'], archive_sha256=audit['archive_sha256'],
                   members=len(audit['members']), deployed_states=audit['deployed_states'],
                   training_loop_seconds=report.get('training_loop_seconds'))
        if self.report['stages'][key].get('status') != 'REUSED_SEALED':
            self.stage(key, status='COMPLETE')
        ref = HISTORICAL_8_12.get(spec['hist']) if self.full_release() and self.seed == SEED else None
        if ref and 'deployed_point' in ref:     # equal only if every earlier stage and this training were bit-exact
            self.report['historical_comparison'][f'{key}_deployed_point_state'] = compare(audit['deployed_states']['model.pt'], ref['deployed_point'])

    def prev_from_stage(self, key):
        d = self.out / self.spec(key)['dir']
        audit = read_json(d / 'package_audit.json')
        return {'key': key, 'candidate': audit['candidate'], 'dir': d, 'zip': ROOT / audit['archive'],
                'zip_sha256': audit['archive_sha256'], 'runtime': ROOT / audit['runtime'], 'stand_in': False}

    def standin_prev(self, key, spec, d):
        """Smoke only: the historical output of this stage (read-only) as the next stage's input. Full release with the
        recipe seed: teacher_source = the historical run dir. Otherwise (folds, --seed): the smoke's own stage dir
        (its generated plan.json) with the historical full-release package as a weight stand-in (plumbing only)."""
        h = HIST[key]
        archive = ROOT / h.get('zip_used_as_teacher', h['zip'])
        if not archive.exists():          # weight-free tree: the stage-7 package is the (plumbing-only) weight stand-in
            return {'key': key, 'candidate': spec['candidate'], 'dir': d, 'zip': self.s7['archive'],
                    'zip_sha256': self.s7['archive_sha256'], 'runtime': self.s7['runtime'], 'stand_in': True}
        return {'key': key, 'candidate': spec['candidate'], 'dir': ROOT / h['dir'] if self.history else d, 'zip': archive,
                'zip_sha256': sha256(archive), 'runtime': self.extract(key, archive), 'stand_in': True}

    # ---------------------------------------------------------------- smoke verification
    def reference_plan(self, key, spec):
        if self.history and spec['hist']:
            return HIST[spec['hist']]['plan'], 'historical full81 plan'
        if self.fold:
            held = HIST_HELD[self.fold].get('s13_sim' if (key == 's13' and spec['sim']) else key)
            if key == 's12' and spec['sim']:
                held = HIST_HELD[self.fold]['s12']
            return held, f'historical held-surface plan ({self.fold})'
        if key == 's12' and spec['sim'] and self.full_release() and self.seed == SEED:
            return HIST['s12']['plan'], 'LC4B full81 plan (no historical counterpart of LC4BSIMH)'
        if key == 's13' and not spec['sim'] and self.full_release() and self.seed == SEED:
            return HIST['s13_sim']['plan'], 'SIMA full81 plan (same real stream as LC5B; train_cycle4 fields only)'
        return None, None

    def compare_plan(self, key, spec, plan, prev):
        ref_path, label = self.reference_plan(key, spec)
        if ref_path is None or not (ROOT / ref_path).exists():
            self.report['plan_comparison'][key] = {'status': 'SKIPPED (no historical reference for this fold/seed)'}
            return
        ref = read_json(ROOT / ref_path)
        scripts = [(spec['script'], ('m', 'parent'))] + ([(SIM_MODULE, ('manifest',))] if spec['sim'] else [])
        scripts.append(PACKAGER_FIELDS['latcadf' if key == 's8' else 'sd10'])
        fields = sorted({p for script, names in scripts for p in plan_fields_read(script, names)})
        rows, unexpected = [], []
        mine_parent = read_json(prev['dir'] / 'plan.json') if prev else None
        ref_parent = read_json(ROOT / ref['teacher_source'] / 'plan.json') if 'teacher_source' in ref and (ROOT / ref['teacher_source'] / 'plan.json').exists() else None
        for path in fields:
            root, keys = path[0], path[1:]
            if root == 'parent':
                a, b = dig(mine_parent, keys), dig(ref_parent, keys)
            else:
                full = ALIASES.get(root, ('m',))[1:] + keys
                a, b = dig(plan, full), dig(ref, full)
                path = ('m',) + full
            reason = next((r for p, r in INTENDED.items() if path[:len(p)] == p), None)
            if path == ('m', 'candidate') and a != b and ((key == 's13' and not spec['sim']) or (key == 's12' and spec['sim'])):
                reason = 'different candidate by design (see the reference label)'
            if path == ('m', 'training', 'seed') and a != b and self.seed != SEED:
                reason = '--seed override'
            if path == ('m', 'training', 'sim_replay') and key == 's12' and spec['sim']:
                reason = 'LC4BSIMH adds simulation replay (no historical counterpart)'
            if path == ('m', 'training', 'seed') and a != b and self.fold and self.seed != HELD_FOLD_SEEDS[self.fold]:
                reason = f'recipe seed {self.seed}; the held surface used {HELD_FOLD_SEEDS[self.fold]} (pass --seed to match)'
            if path == ('m', 'candidate') and a != b and self.fold:
                reason = 'held-surface runs are named <X>B_<fold>'
            if (path == ('m', 'reflection_source') and a != b and isinstance(a, str) and isinstance(b, str)
                    and (ROOT / b).is_file() and sha256(ROOT / a) == sha256(ROOT / b)):
                reason = f'same file content (sha256 {sha256(ROOT / a)[:16]}) at another path'
            status = 'equal' if a == b else ('intended' if reason else 'UNEXPECTED')
            if status == 'UNEXPECTED':
                unexpected.append('.'.join(path))
            rows.append({'field': '.'.join(path), 'status': status, 'reason': reason if a != b else None,
                         'value': a if status != 'equal' and not isinstance(a, dict) else None,
                         'reference': b if status != 'equal' and not isinstance(b, dict) else None})
        tdiff = sorted(k for k in set(plan['training']) | set(ref['training']) if plan['training'].get(k, MISSING) != ref['training'].get(k, MISSING))
        sdiff = sorted(k for k in set(plan['sampling']) | set(ref['sampling']) if plan['sampling'].get(k) != ref['sampling'].get(k))
        allowed = {'seed', 'lattice_phase_seed'} if self.seed != (HELD_FOLD_SEEDS.get(self.fold) if self.fold else SEED) else set()
        if key == 's12' and spec['sim']:
            allowed |= {'sim_replay'}
        if key == 's13' and not spec['sim']:
            allowed |= {'sim_replay'}          # reference = SIMA plan
        result = {'reference': ref_path, 'label': label, 'fields_compared': len(rows),
                  'equal': sum(r['status'] == 'equal' for r in rows), 'intended': sum(r['status'] == 'intended' for r in rows),
                  'unexpected': unexpected, 'training_keys_differing': tdiff, 'sampling_keys_differing': sdiff,
                  'rows': rows}
        # Full release: the whole training and sampling dicts must equal the historical plan. Held-surface plans carry
        # descriptive drift (sampling labels, a missing phase_formula text, an old card's record under sampling), so
        # for them only keys that a script reads are fatal; the rest is recorded.
        if self.fold:
            read_t = {p[2] for p in fields if p[:2] == ('m', 'training') and len(p) > 2}
            read_s = {p[2] for p in fields if p[:2] == ('m', 'sampling') and len(p) > 2}
            bad = [k for k in tdiff if k not in allowed and k in read_t] + [k for k in sdiff if k in read_s]
            result['descriptive_differences_ignored'] = {'training': [k for k in tdiff if k not in read_t],
                                                         'sampling': [k for k in sdiff if k not in read_s]}
        else:
            bad = [k for k in tdiff if k not in allowed] + sdiff
        result['status'] = 'OK' if not unexpected and not bad else 'FAILED'
        self.report['plan_comparison'][key] = {k: v for k, v in result.items() if k != 'rows'}
        self.verification(f'plan_{key}', result)
        self.log(f'PLAN {key} vs {ref_path}: {result["equal"]} equal, {result["intended"]} intended, '
                 f'unexpected={unexpected} training_diff={tdiff} sampling_diff={sdiff}')
        if result['status'] != 'OK':
            raise StageError(f'{key} plan differs from {ref_path} beyond the intended differences: {unexpected + bad}')

    def compare_streams(self, key, hist_key, streams, d=None, plan_name=None):
        refs, sim_refs, extra = {}, {}, {}
        if self.full_release() and self.seed == SEED:
            # s13 (LC5BH or SIMAH) shares the offset-200 real stream; LC4BSIMH shares the LC4B (offset 100) stream;
            # the simulation stream is drawn at plan seed+19 whatever the offset (sim_replay_all.py:33)
            ref_key = 's13' if key == 's13' else (hist_key if hist_key in HISTORICAL_8_12 else key)
            refs = {k: v for k, v in HISTORICAL_8_12.get(ref_key, {}).items() if not k.startswith('sim_')}
            if streams.get('simulation_replay'):
                sim_refs = {k: v for k, v in HISTORICAL_8_12['s13_sim'].items() if k.startswith('sim_')}
        elif self.fold and self.seed == HELD_FOLD_SEEDS[self.fold] and self.smoke:
            refs, held_sim = self.held_stream_refs(key)
            if held_sim and streams.get('simulation_replay'):
                extra = self.held_sim_protocol(key, d, plan_name, held_sim, streams['simulation_replay'])
        if not refs:
            return
        mine = dict(streams)
        if streams.get('middle'):
            mine['middle_batch_digest'] = streams['middle']['batch_digest']
        if streams.get('simulation_replay'):
            sim = streams['simulation_replay']
            mine.update(sim_digest=sim['digest'], sim_samples=sim['samples'], sim_windows=sim['windows'])
        refs.update(sim_refs)
        rows = {k: compare(mine.get(k), v) for k, v in refs.items() if k in mine}
        self.report['stream_comparison'][key] = {**{k: v['match'] for k, v in rows.items()}, **extra}
        self.verification(f'streams_{key}', {'replayed': streams, 'reference': refs, 'comparison': rows, 'held_simulation_protocol': extra})
        bad = [k for k, v in rows.items() if not v['match']]
        self.log(f'STREAMS {key}: ' + ', '.join(f'{k}={"match" if v["match"] else "DIFF"}' for k, v in rows.items())
                 + (f' | held sim protocol: {json.dumps(extra)}' if extra else ''))
        if bad and self.smoke:
            raise StageError(f'{key} replayed streams differ from the historical run: {bad}')

    def held_stream_refs(self, key):
        """Held-surface runs record the same stream statistics in report.json (fold smoke with the fold's seed)."""
        spec = self.spec(key)
        if key == 's12' and spec['sim']:
            return {}, {}
        path = HIST_HELD[self.fold].get('s13_sim' if (key == 's13' and spec['sim']) else key)
        report = ROOT / Path(path).parent / 'training_candidate' / 'report.json'
        if not report.exists():
            return {}, {}
        r = read_json(report)
        if key == 's8':
            return {k: r[k] for k in ('phase_digest', 'cadence_digest', 'augmentation_digest', 'phase_counts',
                                      'cadence_counts', 'cadence_trailing_fallbacks')}, {}
        ph = r['phases'][-1]
        refs = {k: ph[k] for k in ('batch_digest', 'augmentation_digest', 'lattice_phase_counts', 'cadence_counts',
                                   'cadence_trailing_fallbacks', 'unique_sampled_windows')}
        return refs, r.get('simulation_replay') or {}

    def held_sim_protocol(self, key, d, plan_name, held, fold_sim):
        """Historical held SIMA runs replayed ALL simulations (incl. the held regimes); this rebuild drops the held
        regimes (stage-1-7 fold rule). Replaying the historical protocol validates the simulation-stream replay; the
        fold-rule stream is expected to differ."""
        inv_all = self.out / 'inputs' / 'inventory_allsims.json'
        if not inv_all.exists():
            write_json(self.out / 'inputs' / 'no_exclusion.json', [])
            self.run('inventory_allsims', [self.py, DRIVER / 'preflight_8_12.py', 'inventory', '--cases-json',
                                           self.out / 'inputs' / 'train_cases.json', '--sim-excluded-json',
                                           self.out / 'inputs' / 'no_exclusion.json', '--out', inv_all], cpu=True)
        target = d / 'preflight_streams_allsims.json'
        self.run(f'{key}_replay_allsims', [self.py, DRIVER / 'preflight_8_12.py', 'replay', '--plan', d / plan_name,
                                           '--inventory', inv_all, '--kind', 'sd10', '--out', target], cpu=True)
        sim = read_json(target)['simulation_replay']
        return {'historical_protocol_all_sims_digest_match': sim['digest'] == held['digest'],
                'historical_protocol_windows': [sim['windows'], held['windows']],
                'fold_rule_windows': fold_sim['windows'], 'fold_rule_digest_differs_by_design': fold_sim['digest'] != held['digest']}

    def parity(self, key, spec, plan_path):
        """Lifted packager on the historical training outputs of this stage vs the historical package ZIP."""
        hk = spec['hist']
        h = HIST[hk]
        kind = 'latcadf' if key == 's8' else 'sd10'
        ref_zip = ROOT / h.get('submission_zip', h['zip'])
        hist_plan = ROOT / h['plan']
        t = ROOT / h['dir'] / 'training_candidate'
        variants = {'rebuilt_plan': [plan_path, None, None]}
        if key == 's8':
            variants['historical_template'] = [hist_plan, None, None]      # C1017H payload templates (historical B2)
        else:
            teacher = HIST[{'s9': 's8', 's10': 's9', 's11': 's10', 's12': 's11', 's13': 's12'}[key]]
            override = ROOT / teacher['zip_used_as_teacher'] if 'zip_used_as_teacher' in teacher else None
            variants['historical_plan'] = [hist_plan, override, ROOT / h['dir'] / 'training_script.txt']
        result = {'reference_zip': rel(ref_zip), 'reference_checkpoint': h['checkpoint'], 'variants': {}}
        for name, (plan, template, script) in variants.items():
            out = self.out / 'parity' / key / name
            argv = [self.py, DRIVER / 'package_8_12.py', kind, '--plan', plan, '--trained-plan', hist_plan,
                    '--training-dir', t, '--out', out]
            if template:
                argv += ['--template-zip', template]
            if script:
                argv += ['--script-spec', script]
            self.run(f'parity_{key}_{name}', argv, cpu=True)
            audit = read_json(out / 'package_audit.json')
            target = self.out / 'verification' / f'package_{key}_{name}.json'
            self.run(f'parity_{key}_{name}_compare', [self.py, DRIVER / 'preflight_8_12.py', 'compare', '--rebuilt',
                                                      ROOT / audit['archive'], '--reference', ref_zip, '--out', target], cpu=True)
            cmp = read_json(target)
            cmp['deployed_checkpoint_identical'] = audit['deployed_checkpoint_sha256'] == sha256(ROOT / h['checkpoint'])
            differences, unexplained = [], []
            for row in cmp['rows']:
                if row['status'] != 'identical':
                    why = self.explain(key, name, audit['candidate'], row)
                    row['reason'] = PACKAGE_REASONS.get(why)
                    differences.append({'member': row['member'], 'status': row['status'], 'reason': why})
                    if why is None:
                        unexplained.append(row['member'])
            if not cmp['deployed_checkpoint_identical']:
                unexplained.append('deployed checkpoint')
            write_json(target, cmp)
            result['variants'][name] = {'plan': rel(plan), 'template_archive': audit['template_archive']['path'],
                                        'members_rebuilt': cmp['members_rebuilt'], 'members_reference': cmp['members_reference'],
                                        'identical': cmp['identical'], 'differences': differences, 'unexplained': unexplained,
                                        'common_member_order_equal': cmp['common_member_order_equal'],
                                        'deployed_checkpoint_identical': cmp['deployed_checkpoint_identical'], 'file': rel(target)}
            self.log(f'PARITY {key}/{name}: {cmp["identical"]}/{cmp["members_reference"]} reference members byte-identical; '
                     f'differences {[(d["member"], d["reason"]) for d in differences]}; checkpoint identical={cmp["deployed_checkpoint_identical"]}')
            if unexplained or not cmp['common_member_order_equal']:
                raise StageError(f'{key}/{name} package differs from {rel(ref_zip)} without a known reason: {unexplained}')
        result['reasons'] = PACKAGE_REASONS
        self.report['package_parity'][key] = result

    def explain(self, key, variant, candidate, row):
        member, status = row['member'], row['status']
        if key == 's9' and (status == 'renamed_identical' or member in ('training_recipe/sd10_continuation.json',
                                                                         'training_recipe/sd10_train.py.txt')):
            return 'rename'
        if variant == 'rebuilt_plan' and member == f'training_recipe/{candidate}_continuation.json':
            return 'continuation'
        if key == 's12' and member == 'training_recipe/LC4BH_train.py.txt':
            return 'b5'
        if key == 's13' and member == 'training_recipe/SIMAH_sim_replay_all.py.txt':
            return 'sim_module'
        if (key == 's8' and variant == 'rebuilt_plan' and member in WEIGHTS and row.get('state_dict_equal')
                and row.get('payload_keys_only_reference') == ['reflection_fraction', 'reflection_point_lr']
                and not row.get('payload_keys_only_rebuilt')):
            if not row.get('payload_keys_differing'):
                return 'b2_payload'
            if not self.s7['stand_in'] and set(row['payload_keys_differing']) <= PROVENANCE_KEYS:
                return 'b2_payload_rebuilt_s7'
        return None

    # ---------------------------------------------------------------- main flow
    def execute(self):
        self.prologue()
        self.resolve_inputs()
        prev = self.stage8()
        for key in self.keys[1:]:
            prev = self.stage_sd10(key, prev)
        if self.smoke:
            return None
        return ROOT / prev['runtime'] if prev else None

    def finish(self, status, runtime=None, error=None):
        self.report['status'] = status
        self.report['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
        if runtime is not None:
            self.report['final_runtime'] = rel(runtime)
        if error is not None:
            self.report['error'] = error
        mism = [k for k, v in self.report['historical_comparison'].items() if isinstance(v, dict) and v.get('match') is False]
        self.report['historical_mismatches_recorded'] = mism
        est = [s['estimate_seconds'] for s in self.report['gpu_steps']]
        if est:
            self.report['gpu_estimate_total_seconds'] = {
                'unshared': sum(e['unshared'] for e in est),
                'historical_logged_shared': sum(e['historical_logged_shared'] or 0 for e in est),
                'stages_without_full81_history': [s['stage'] for s in self.report['gpu_steps'] if s['estimate_seconds']['historical_logged_shared'] is None]}
        if self.owned:
            write_json(self.out / 'rebuild_report.json', self.report)
        self.log(f'{status} elapsed={self.report["elapsed_seconds"]} s; recorded historical mismatches: {mism or "none"}')


def summarize_smoke(driver):
    r = driver.report
    driver.log('SMOKE SUMMARY (CPU only; nothing ran on the GPU)')
    for name, values in r['stages'].items():
        driver.log(f'  stage {name}: {values.get("status", "")}')
    for key, value in r['plan_comparison'].items():
        driver.log(f'  plan {key}: {json.dumps({k: value.get(k) for k in ("status", "reference", "equal", "intended", "unexpected", "training_keys_differing")})}')
    for key, value in r['stream_comparison'].items():
        driver.log(f'  streams {key}: {json.dumps(value)}')
    for key, value in r['package_parity'].items():
        for name, v in value['variants'].items():
            driver.log(f'  package {key}/{name}: {v["identical"]}/{v["members_reference"]} byte-identical; differences '
                       f'{[(d["member"], d["reason"]) for d in v["differences"]]}; checkpoint identical {v["deployed_checkpoint_identical"]}')
    for step in r['gpu_steps']:
        e = step['estimate_seconds']
        driver.log(f'  GPU needed: {step["stage"]} ({step["kind"]}, {step["steps"]} updates) ~{e["unshared"]} s unshared; '
                   f'historical {e["historical_logged_shared"]} s shared')
    if r.get('gpu_estimate_total_seconds'):
        t = r['gpu_estimate_total_seconds']
        driver.log(f'  GPU total (one A5500): ~{t["unshared"]} s unshared; historical {t["historical_logged_shared"]} s shared '
                   f'(no full81 history: {t["stages_without_full81_history"]})')


def cleanup_smoke(driver):
    """Keep driver.log, rebuild_report.json, holdout.json and verification/; remove stand-ins, parity packages, plans."""
    for path in sorted(driver.out.iterdir()):
        if path.name not in ('driver.log', 'rebuild_report.json', 'verification', 'holdout.json'):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='new directory under runs/FINAL_REBUILD/')
    parser.add_argument('--stage7', help='COMPLETE rebuild_1_7.py output directory (required unless --smoke)')
    parser.add_argument('--holdout-re', help="comma list of Reynolds numbers and/or one '>=X' term (as rebuild_1_7.py)")
    parser.add_argument('--holdout-aoa', help='comma list of angles of attack (as rebuild_1_7.py)')
    parser.add_argument('--seed', type=int, default=SEED, help=f'recipe seed of stages 8-13 (default {SEED})')
    parser.add_argument('--extra-cycle', action='store_true', help='add stage 13 (LC5-style continuation, offset 200)')
    parser.add_argument('--sim-replay', choices=['none', 'all'], default='none',
                        help="'all': the final cycle uses runs/SIMALL/train_sim.py with training.sim_replay='all'")
    parser.add_argument('--smoke', action='store_true', help='CPU-only verification; GPU stages are dry-run')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted run in --out (sealed stages reused)')
    parser.add_argument('--keep-smoke-outputs', action='store_true', help='keep all smoke outputs (default: logs, report, verification/)')
    parser.add_argument('--python', help='interpreter for all stage processes (default: this interpreter)')
    parser.add_argument('--stop-before-gpu', action='store_true',
                        help='rebuild mode: do plans and CPU checks, stop before the first GPU training launch (resume later)')
    args = parser.parse_args(argv)
    driver = Driver(args)
    status, runtime, error = 'FAILED', None, None
    try:
        runtime = driver.execute()
        status = 'SMOKE_COMPLETE' if driver.smoke else 'COMPLETE'
    except StopBeforeGpu as stop:
        status = 'STOPPED_BEFORE_GPU'
        driver.log(f'STOP before the GPU training of {stop}; continue with --resume (without --stop-before-gpu)')
    except Exception as exc:          # recorded in rebuild_report.json and logs/driver_traceback.txt
        error = f'{type(exc).__name__}: {exc}'
        driver.log('ERROR ' + error)
        if driver.owned:
            (driver.out / 'logs').mkdir(exist_ok=True)
            (driver.out / 'logs' / 'driver_traceback.txt').write_text(traceback.format_exc())
    finally:
        if driver.owned:
            driver.finish(status, runtime, error)
            if driver.smoke:
                summarize_smoke(driver)
                if not args.keep_smoke_outputs and status != 'FAILED':
                    cleanup_smoke(driver)
    return 0 if status != 'FAILED' else 1


if __name__ == '__main__':
    sys.exit(main())
