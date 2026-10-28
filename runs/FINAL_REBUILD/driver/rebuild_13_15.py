"""FINAL_REBUILD driver, stages 13-15 + final packaging: stage-12 output (LC4BH-equivalent, rebuild_8_12.py) ->
SIMSAT2D4FH-equivalent deployable ZIP (4-block student + fixed network batch 5 + host-allocation block).

Full documentation: runs/FINAL_REBUILD/driver/README_13_15.md (usage, stages, checks, remaining dependencies, GPU time).

Usage (any cwd; relative paths are relative to the project root; interpreter .venv310/bin/python)
  rebuild  rebuild_13_15.py --stage12 runs/FINAL_REBUILD/<R>_8_12 --out runs/FINAL_REBUILD/<R>_13_15
           [--loss-mode default|old] [--focus-fraction F | --focus-ratio R] [--s15-offset N] [--stop-before-gpu] [--resume]
  smoke    rebuild_13_15.py --smoke --out runs/FINAL_REBUILD/smoke_13_15/<name> [--stage12 DIR]
           [--holdout-re LIST] [--holdout-aoa LIST] [--seed N] [same S15 options] [--keep-smoke-outputs]
Stages (historical trainers run unchanged via rebind_13_15.py; plans generated here; packaging: package_13_15.py)
  s13 SIMSATH      runs/TTAKD/train_simat.py + runs/SIMALL/sim_replay_all.py, teacher = stage 12, offset 200
  s14 SIMSAT2H     same trainer, teacher = s13, offset 900
  s15 SIMSAT2D4FH  runs/TTAKD/train_simati_focus.py + runs/SIMALL/sim_replay_focus.py, teacher = s14, 4-block student
                   [0,4,5,9], distillation S6 + fine-tune 2*S6, interval KD, focus fraction .5, offset 1800
                   (--loss-mode old: train_simati_focus_old.py = E11 SIMSAT2D4FOH; --focus-fraction F: other per-batch
                   focus count round(8F); --focus-ratio R: train_simati_focus2*.py + sim_replay_focus2.py)
  final            F5 (make_fixed5.py) + MA (make_malloc.py) member transforms of the s15 package -> deliverable ZIP,
                   CPU runtime smoke (runtime_smoke_13_15.py)
Per stage: plan -> CPU checks (source pins, the trainer's own --cpu-preflight, CPU prediction of the initial state
hashes, CPU replay of every seeded stream) -> GPU training -> report/job.json must equal the CPU predictions -> package ->
seal. Stage dir <out>/sN_<candidate>/: plan.json, training_script.txt, cpu_preflight/, preflight_states.json,
preflight_streams.json, training_candidate/, runtime/, <CAND>_REBUILT.zip, <CAND>_deployed_checkpoint.pt,
package_audit.json; final: <out>/final_<cand>f5ma/{<C>F5MA_FIXED5_REBUILT.zip, runtime/, final_audit.json,
runtime_smoke.json}; <out>/rebuild_report.json, driver.log, logs/, seals/, verification/.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DRIVER, ENV, EXCLUDED_REAL, REBUILD_BASE, RETAINED_REAL, ROOT, compare, read_json, regime, rel, release_names, sha256, write_json  # noqa: E402
from common_8_12 import HELD_FOLD_SEEDS, RUNTIME_C973, RUNTIME_C973_DIR, SEED, plan_fields_read, sd10_training  # noqa: E402
from common_13_15 import (CODE_PINS_13_15, DEFAULT_FOCUS_FRACTION, DEFAULT_S15_OFFSET, GPU_ESTIMATES_13_15, GPU_FREE_MIB_REQUIRED, HIST_13_15, HIST_FINAL,  # noqa: E402
                          HIST_HELD_13_15, HIST_S15_OLD, HISTORICAL_13_15, LIFTED_FROM_13_15, PACKAGING_PINS, SD10_REFLECTION,
                          SIM_OF, STAGES_13_15, d4_training, estimate_seconds, s15_variant, script_closure_13_15, simsat_training)
import c675_ext  # noqa: E402
import rebuild_8_12  # noqa: E402
from rebuild_8_12 import ALIASES, MISSING, StageError, StopBeforeGpu, cleanup_smoke, compute_holdout, dig  # noqa: E402
from rebuild_1_7 import parse_aoa, parse_re  # noqa: E402

DRIVER_FILES = ['common.py', 'common_8_12.py', 'common_13_15.py', 'rebuild_13_15.py', 'rebind_13_15.py', 'preflight_13_15.py',
                'package_13_15.py', 'runtime_smoke_13_15.py', 'rebind_8_12.py', 'preflight_8_12.py', 'package_8_12.py',
                'rebuild_8_12.py', 'rebuild_1_7.py', 'c675_ext.py', 'preflight.py']
PACKAGER = 'runs/SD10/finalize_sd10.py'          # historical packager whose plan reads are compared (fields only)
INTENDED = {
    ('m', 'source_pins'): 'B1: code closure + previous rebuilt runtime; historical pins held non-lineage C1017R/C1031R/C728R artifacts',
    ('m', 'runtime'): 'runtime/ of the previous stage of this rebuild (historical: a teacher_runtime extraction in the run dir)',
    ('m', 'teacher_source'): 'previous stage directory of this rebuild',
    ('m', 'teacher_archive'): 'previous stage package of this rebuild (payload templates, carried training_recipe/)',
    ('m', 'runtime_baseline'): 'B2: vendored runtime_code_c973h/ instead of the weight-bearing C973H ZIP',
}
PACKAGE_REASONS = {
    'continuation': 'intended: <CAND>_continuation.json embeds the rebuilt plan (rebuild paths, code+runtime source_pins, '
                    'vendored runtime_baseline) instead of the historical plan',
}
PHASE_KEYS = ('batch_digest', 'augmentation_digest', 'lattice_phase_counts', 'cadence_counts', 'cadence_trailing_fallbacks',
              'unique_sampled_windows', 'source_samples')
SIM_KEYS = {'sim_digest': 'digest', 'sim_samples': 'samples', 'sim_unique_windows': 'unique_windows', 'sim_windows': 'windows',
            'sim_calls': 'calls', 'sim_first_sample_ids': 'first_sample_ids'}


def flatten_streams(streams):
    flat = {}
    for ph in streams['phases']:
        for k in PHASE_KEYS:
            flat[f"{ph['phase']}.{k}"] = ph[k]
    for step, c in streams['checkpoints'].items():
        for k in ('batch_digest', 'augmentation_digest'):
            flat[f'checkpoint.{step}.{k}'] = c[k]
    sim = streams.get('simulation_replay')
    if sim:
        flat.update({k: sim[v] for k, v in SIM_KEYS.items()})
    return flat


class Driver(rebuild_8_12.Driver):
    """Reuses the stage 8-12 driver utilities (logging, child runs, seals, plan writing, holdout records, GPU launch
    gate, finish); everything stage specific is defined here."""

    def __init__(self, args):   # noqa: D401  (the 8-12 __init__ validates --stage7; not called)
        out = Path(args.out)
        out = (out if out.is_absolute() else ROOT / out).resolve()
        if REBUILD_BASE not in out.parents or out == DRIVER or DRIVER in out.parents:
            raise SystemExit('--out must be a new directory under runs/FINAL_REBUILD/ (not driver/)')
        if args.smoke and args.resume:
            raise SystemExit('--smoke cannot be combined with --resume')
        if out.exists() != bool(args.resume):
            raise SystemExit('--out must not exist (or must exist with --resume): ' + str(out))
        if not args.smoke and not args.stage12:
            raise SystemExit('--stage12 (a COMPLETE rebuild_8_12.py output) is required unless --smoke')
        if args.focus_fraction is not None and args.focus_ratio is not None:
            raise SystemExit('--focus-fraction and --focus-ratio are mutually exclusive')
        if args.focus_fraction is not None and not 0.0 < args.focus_fraction < 1.0:
            raise SystemExit('--focus-fraction must be in (0, 1) (sim_replay_focus.py:37)')
        if args.focus_ratio is not None and args.focus_ratio < 1.0:
            raise SystemExit('--focus-ratio must be >= 1 (sim_replay_focus2.py:37)')
        if args.s15_offset < 0:
            raise SystemExit('--s15-offset must be >= 0')
        self.stage12_dir = None
        if args.stage12:
            s12 = Path(args.stage12)
            self.stage12_dir = (s12 if s12.is_absolute() else ROOT / s12).resolve(strict=True)
        self.out, self.args, self.smoke = out, args, args.smoke
        self.owned = False
        self.py = args.python or sys.executable
        self.flags_given = bool(args.holdout_re or args.holdout_aoa)
        self.re_values, self.re_ge = parse_re(args.holdout_re)
        self.aoa_values = parse_aoa(args.holdout_aoa)
        names = release_names('train_sim') + release_names('train_real')
        unknown = ([v for v in self.re_values if v not in {regime(n)[0] for n in names}]
                   + [v for v in self.aoa_values if v not in {regime(n)[1] for n in names}])
        if unknown:
            raise SystemExit(f'holdout values not in the release: {unknown}')
        self.seed = args.seed
        self.loss_mode = args.loss_mode
        self.focus = ('ratio', float(args.focus_ratio)) if args.focus_ratio is not None else (
            'fraction', float(args.focus_fraction if args.focus_fraction is not None else DEFAULT_FOCUS_FRACTION))
        self.s15_offset = int(args.s15_offset)
        self.keys = ['s13', 's14', 's15']
        self.history = False
        self.report = {'status': 'RUNNING', 'mode': 'smoke_cpu' if self.smoke else 'rebuild', 'out': None,
                       'options': {'loss_mode': self.loss_mode, 'focus': list(self.focus), 's15_offset': self.s15_offset},
                       'stages': {}, 'plan_comparison': {}, 'stream_comparison': {}, 'state_comparison': {},
                       'package_parity': {}, 'final_parity': {}, 'historical_comparison': {}, 'gpu_steps': [], 'commands': []}
        self.started = time.monotonic()

    # ---------------------------------------------------------------- utilities
    def env(self, cpu=False):
        env = super().env(cpu)
        if not (cpu or self.smoke):     # historical GPU launches of these trainers (launch_simsat.sh, chain_e6.sh)
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        return env

    def gpu_step(self, key, kind, steps, argv):
        est = estimate_seconds(kind, steps, key)
        self.report['gpu_steps'].append({'stage': key, 'kind': kind, 'steps': steps, 'command': [str(a) for a in argv],
                                         'estimate_seconds': {'unshared': est[0], 'historical_logged_shared': est[1]},
                                         'note': GPU_ESTIMATES_13_15[kind]['note']})
        self.log(f'DRY-RUN {key}: GPU step not run (one A5500: ~{est[0]} s unshared; historical wall {est[1]} s shared)')

    def gpu_launch(self, key, argv):
        if self.args.stop_before_gpu:
            self.stage(key, status='STOPPED_BEFORE_GPU', gpu_command=[str(a) for a in argv])
            raise StopBeforeGpu(key)
        self.gpu_probe(key)
        self.run(f'{key}_train', argv)

    def gpu_probe(self, key):
        """Right before a launch (never in smoke / --stop-before-gpu): CUDA visible and enough free device memory (the
        historical launch gates), so that a shared or hidden GPU fails here and not inside the trainer."""
        probe = ("import json,torch;ok=torch.cuda.is_available();free,total=torch.cuda.mem_get_info() if ok else (0,0);"
                 "print(json.dumps({'cuda':ok,'device':torch.cuda.get_device_name(0) if ok else None,'free_mib':free>>20,'total_mib':total>>20}))")
        info = self.last_json(self.run(f'{key}_gpu_probe', [self.py, '-c', probe]))
        need = GPU_FREE_MIB_REQUIRED[STAGES_13_15[key]['kind']]
        self.stage(key, gpu_probe=dict(info, required_free_mib=need))
        if not info['cuda']:
            raise StageError(f'{key}: CUDA is not available to the training process (CUDA_VISIBLE_DEVICES?)')
        if info['free_mib'] < need:
            raise StageError(f'{key}: {info["free_mib"]} MiB free on {info["device"]}, {need} MiB required (the trainer caps its '
                             'allocator at 4 GiB); free the GPU, then --resume')

    def spec(self, key):
        s = dict(STAGES_13_15[key], key=key)
        if key == 's15':
            s['candidate'], s['script'], s['hist'] = s15_variant(self.loss_mode, self.focus, self.s15_offset)
            s['offset'] = self.s15_offset
        s['sim_module'] = SIM_OF[s['script']]
        s['dir'] = f"{key}_{s['candidate'].lower()}"
        return s

    def training(self, key):
        if key == 's15':
            return d4_training(self.base, self.seed, self.loss_mode, self.focus, self.s15_offset)
        return simsat_training(self.base, self.seed, STAGES_13_15[key]['offset'])

    # ---------------------------------------------------------------- prologue
    def prologue(self):
        manifest = {'stage12': rel(self.stage12_dir) if self.stage12_dir else None, 'holdout_re': self.re_values,
                    'holdout_re_ge': self.re_ge, 'holdout_aoa': self.aoa_values, 'seed': self.seed,
                    'loss_mode': self.loss_mode, 'focus': list(self.focus), 's15_offset': self.s15_offset, 'smoke': self.smoke,
                    'python': self.py}
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
        self.log(f'START {self.report["mode"]} out={rel(self.out)} loss_mode={self.loss_mode} focus={self.focus} '
                 f's15_offset={self.s15_offset} seed={"inherited/default" if self.seed is None else self.seed}')
        for path, expected in {**CODE_PINS_13_15, **PACKAGING_PINS}.items():          # code pins (asserted)
            if sha256(ROOT / path) != expected:
                raise StageError('code pin mismatch: ' + path)
        for script in SIM_OF:                                                          # closures unchanged (asserted)
            extra = set(script_closure_13_15(script)) - set(CODE_PINS_13_15)
            if extra:
                raise StageError(f'import closure of {script} gained unpinned files: {sorted(extra)}')
        vendored = {str(p.relative_to(RUNTIME_C973_DIR)): sha256(p) for p in sorted(RUNTIME_C973_DIR.rglob('*')) if p.is_file()}
        if vendored != RUNTIME_C973:
            raise StageError('vendored runtime_code_c973h differs from its pins')
        names = release_names('train_real')                                            # data layout (asserted)
        assert len(names) == 81 and RETAINED_REAL in names and EXCLUDED_REAL not in names
        assert len(release_names('train_sim')) == 100  # Decision-phase data omits train_real/7575_0.h5 (always excluded)
        if not (self.out / 'environment.json').exists():
            probe = ("import json,platform,sys,torch,numpy,h5py;print(json.dumps({'python':sys.version.split()[0],"
                     "'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),"
                     "'numpy':numpy.__version__,'h5py':h5py.__version__,'platform':platform.platform()}))")
            log = self.run('environment_probe', [self.py, '-c', probe], cpu=True)
            files = sorted({**CODE_PINS_13_15, **PACKAGING_PINS}) + [str((DRIVER / f).relative_to(ROOT)) for f in DRIVER_FILES]
            write_json(self.out / 'environment.json', {
                'software': self.last_json(log), 'env': ENV, 'gpu_training_env_extra': {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
                'code_sha256': {f: sha256(ROOT / f) for f in files},
                'vendored_runtime_code': {'dir': rel(RUNTIME_C973_DIR), 'sha256': RUNTIME_C973},
                'lifted_packaging_code_recorded': LIFTED_FROM_13_15})

    # ---------------------------------------------------------------- stage-12 input and holdout
    def resolve_inputs(self):
        if self.stage12_dir is not None:
            s12 = self.load_stage12()
        else:
            if self.seed is None:
                self.seed = SEED
            s12 = {'holdout': compute_holdout(self.re_values, self.re_ge, self.aoa_values)}
        self.holdout = s12['holdout']
        cases = self.holdout['cases']
        self.train_cases, self.eval_cases = cases['real_train'], cases['real_evaluation']
        self.sim_excluded = cases['sim_excluded']
        write_json(self.out / 'holdout.json', self.holdout)
        write_json(self.out / 'inputs' / 'train_cases.json', self.train_cases)
        write_json(self.out / 'inputs' / 'sim_excluded.json', self.sim_excluded)
        if not (self.out / 'inputs' / 'inventory.json').exists():
            self.run('inventory', [self.py, DRIVER / 'preflight_8_12.py', 'inventory', '--cases-json',
                                   self.out / 'inputs' / 'train_cases.json', '--sim-excluded-json',
                                   self.out / 'inputs' / 'sim_excluded.json', '--out', self.out / 'inputs' / 'inventory.json'], cpu=True)
        self.inv = read_json(self.out / 'inputs' / 'inventory.json')
        self.base, self.windows = self.inv['base_steps'], self.inv['windows']
        self.fold = next((t for t, kw in (('upperRe', {'holdout_re_ge': 19050}), ('AoA15', {'holdout_aoa': 15}),
                                          ('AoA20', {'holdout_aoa': 20}))
                          if self.holdout['mode'] != 'none' and c675_ext.STOCK_SPLIT(
                              release_names('train_sim'), release_names('train_real'), **kw)['real_train'] == self.train_cases), None)
        if s12.get('dir') is None:
            s12 = self.standin_stage12()
        elif (s12['plan']['training']['steps'], s12['plan']['sampling']['windows']) != (2 * self.base, self.windows):
            raise StageError('stage-12 plan steps/windows differ from the inventory rule (steps = 2*S6)')
        self.s12 = s12
        self.history = self.smoke and self.full_release() and self.seed == SEED and all(
            (ROOT / h[k]).exists() for h in HIST_13_15.values() for k in ('dir', 'plan', 'zip'))
        self.log(f'HOLDOUT mode={self.holdout["mode"]} counts={self.holdout["counts"]} windows={self.windows} base(S6)={self.base} '
                 f'held_surface={self.fold} stage12={"stand-in" if s12["stand_in"] else rel(s12["dir"])} seed={self.seed}')
        self.report['options']['seed'] = self.seed
        self.stage('inputs', status='RESOLVED', holdout_mode=self.holdout['mode'], counts=self.holdout['counts'], windows=self.windows, base_steps=self.base,
                   sim_cases=len(self.inv['sim']), sim_windows=self.inv['sim_windows'], stage12=None if s12['stand_in'] else rel(s12['dir']),
                   stage12_runtime=rel(s12['runtime']), stage12_archive=rel(s12['zip']), stand_in=s12['stand_in'],
                   held_surface_match=self.fold, history_available=self.history, seed=self.seed)

    def load_stage12(self):
        d = self.stage12_dir
        report = read_json(d / 'rebuild_report.json')
        if report.get('status') != 'COMPLETE' or report.get('mode') != 'rebuild':
            raise StageError(f'{rel(d)} is not a COMPLETE rebuild_8_12.py output (status/mode)')
        if (d / 'MOCK_STAGE12.json').exists():
            self.log(f'WARNING {rel(d)} is a MOCK stage-12 directory (test fixture built from historical files)')
            self.report['stage12_is_mock'] = True
        m12 = read_json(d / 'driver_manifest.json')
        if m12.get('smoke'):
            raise StageError('the stage-12 directory is a smoke output')
        for seal in sorted((d / 'seals').glob('*.json')):                              # every stage 8-12 seal
            self.verify_seal(seal)
        if not (d / 'seals' / 's12.json').exists():
            raise StageError('stage-12 seal seals/s12.json missing')
        s = d / 's12_lc4bh'
        audit = read_json(s / 'package_audit.json')
        if audit['candidate'] != 'LC4BH':
            raise StageError(f'stage 12 of {rel(d)} is {audit["candidate"]}, not LC4BH (was rebuild_8_12 run with --sim-replay all?)')
        runtime, archive = ROOT / audit['runtime'], ROOT / audit['archive']
        if sha256(archive) != audit['archive_sha256']:
            raise StageError('stage-12 archive differs from its sealed package_audit.json')
        sealed = set(read_json(d / 'seals' / 's12.json')['outputs'])
        plan = read_json(s / 'plan.json')
        steps = plan['training']['steps']
        needed = [s / 'plan.json', s / 'training_candidate' / f'resume_{steps}.pt', archive, *[p for p in runtime.rglob('*') if p.is_file()]]
        missing = [rel(p) for p in needed if rel(p) not in sealed]
        if missing:
            raise StageError(f'stage-12 outputs used here are not sealed: {missing[:3]}')
        holdout12 = read_json(d / 'holdout.json')
        req = holdout12['requested']
        re_values, re_ge, aoa_values = req['holdout_re_values'], req['holdout_re_ge'], req['holdout_aoa_values']
        if self.flags_given and (sorted(self.re_values), self.re_ge, sorted(self.aoa_values)) != (sorted(re_values), re_ge, sorted(aoa_values)):
            raise StageError('--holdout flags differ from the stage-12 run (the fold is inherited from it)')
        self.re_values, self.re_ge, self.aoa_values = re_values, re_ge, aoa_values
        mine = compute_holdout(re_values, re_ge, aoa_values)
        if any(holdout12[k] != mine[k] for k in ('mode', 'counts', 'cases')):
            raise StageError('stage-12 holdout.json differs from the recomputed partition')
        if plan['train_cases'] != mine['cases']['real_train'] or plan['eval_cases'] != mine['cases']['real_evaluation']:
            raise StageError('stage-12 plan train/eval cases differ from the holdout partition')
        seed = plan['training']['seed']
        if self.seed is not None and self.seed != seed:
            raise StageError(f'--seed {self.seed} differs from the stage-12 recipe seed {seed} (inherited)')
        self.seed = seed
        self.report['stage12_provenance'] = {'dir': rel(d), 'stage7': m12.get('stage7'), 'seed': seed, 'sim_replay': m12.get('sim_replay'),
                                             'extra_cycle': m12.get('extra_cycle'), 'seals_verified': sorted(p.name for p in (d / 'seals').glob('*.json'))}
        return {'key': 's12', 'candidate': 'LC4BH', 'dir': s, 'zip': archive, 'zip_sha256': audit['archive_sha256'],
                'runtime': runtime, 'stand_in': False, 'historical': False, 'holdout': mine, 'plan': plan,
                'seal': d / 'seals' / 's12.json', 'stage12_manifest': m12}

    def standin_stage12(self):
        """Smoke only. Full release + recipe seed: the historical LC4B_full81 run (read-only) with its package extracted.
        Otherwise (folds, --seed): a synthesized stage-12 plan (this fold's cases, LC4B recipe) in standins/s12 with the
        historical full-release LC4BH package as the weight stand-in (plumbing only)."""
        h = HIST_13_15['s12']
        if not (ROOT / h['zip']).exists():
            raise StageError(f'--smoke without --stage12 needs the historical {h["zip"]} as a read-only stand-in')
        runtime = self.extract('s12', ROOT / h['zip'])
        rec = {'key': 's12', 'candidate': 'LC4BH', 'zip': ROOT / h['zip'], 'zip_sha256': sha256(ROOT / h['zip']),
               'runtime': runtime, 'stand_in': True, 'holdout': self.holdout, 'seal': None}
        if self.full_release() and self.seed == SEED and (ROOT / h['plan']).exists():
            rec.update(dir=ROOT / h['dir'], plan=read_json(ROOT / h['plan']), historical=True)
        else:
            d = self.out / 'standins' / 's12'
            plan = {'candidate': 'LC4BH', 'stand_in': 'synthesized by rebuild_13_15.py --smoke: this fold, LC4B recipe; '
                                                      'weights = historical full-release LC4BH (plumbing only)',
                    **self.common_fields(), 'training': sd10_training(self.base, self.seed, 100, False), 'middle_step': self.base}
            write_json(d / 'plan.json', plan)
            rec.update(dir=d, plan=plan, historical=False)
        return rec

    # ---------------------------------------------------------------- plans
    def pins(self, script, runtime, extra=()):
        files = script_closure_13_15(script)
        missing = [f for f in files if f not in CODE_PINS_13_15]
        if missing:
            raise StageError(f'unpinned closure files: {missing}')
        pins = {f: CODE_PINS_13_15[f] for f in files}
        members = sorted(p for p in Path(runtime).rglob('*') if p.is_file())
        if any('__pycache__' in p.parts for p in members):
            raise StageError(f'{rel(runtime)} contains __pycache__ (a runtime must hold package members only)')
        pins.update({rel(p): sha256(p) for p in members})
        return pins

    def make_plan(self, key, spec, prev):
        tr = self.training(key)
        middle = tr['warmup_steps'] + tr['finetune_steps'] // 2          # resume checkpoint of the fine-tune midpoint
        plan = {'candidate': spec['candidate'], 'generated_by': rel(DRIVER / 'rebuild_13_15.py'),
                'policy': 'SIMSAT/SIMSAT2/SIMSAT2D4F recipes inlined (runs/SIMSAT_full81, runs/SIMSAT2_full81, runs/SIMSAT2D4F_full81 '
                          'plans; chain_simsat2d4.sh + chain_e6.sh derivation): fine-tune 2*S6 (+ distillation S6 for the '
                          '4-block student), middle = distillation + fine-tune/2, EMA .999**(6000/S6); teacher = previous rebuilt stage',
                **self.common_fields(), 'training': tr, 'middle_step': middle,
                'runtime': rel(prev['runtime']), 'teacher_source': rel(prev['dir']),
                'teacher_archive': {'path': rel(prev['zip']), 'sha256': prev['zip_sha256']},
                'reflection_source': SD10_REFLECTION, 'runtime_baseline': self.runtime_baseline(),
                'source_pins': self.pins(spec['script'], prev['runtime']), 'holdout': self.holdout_record(),
                'sim_replay_scope': {'policy': 'all', 'module': spec['sim_module'], 'excluded_by_holdout': self.sim_excluded,
                                     'cases': len(self.inv['sim']), 'windows': self.inv['sim_windows']},
                'stage_note': spec['note'],
                'lineage': {'previous': prev['key'], 'previous_candidate': prev['candidate'], 'stand_in': prev['stand_in']}}
        if key == 's15' and (self.loss_mode != 'default' or self.focus != ('fraction', DEFAULT_FOCUS_FRACTION)
                             or self.s15_offset != DEFAULT_S15_OFFSET):
            plan['variant'] = {'loss_mode': self.loss_mode, 'focus': list(self.focus), 'stream_seed_offset': self.s15_offset,
                               'historical_full81': 'E11 SIMSAT2D4FOH (runs/SIMSAT2D4FO_full81)' if spec['hist'] == 's15_old' else None}
        return plan

    # ---------------------------------------------------------------- child commands
    def rebind_argv(self, spec, d, check=None, preflight=False):
        argv = [self.py, DRIVER / 'rebind_13_15.py', '--script', spec['script'], '--sha256', CODE_PINS_13_15[spec['script']],
                '--stage-dir', d]
        if self.sim_excluded:
            argv += ['--sim-module', spec['sim_module'], '--sim-module-sha256', CODE_PINS_13_15[spec['sim_module']],
                     '--sim-exclude-json', self.out / 'inputs' / 'sim_excluded.json']
        if check:
            argv += ['--check', check]
        if preflight:
            argv += ['--cpu-preflight']
        return argv

    def cpu_checks(self, key, spec, d):
        pre = d / 'cpu_preflight' / 'report.json'
        if (d / 'cpu_preflight').exists() and not (pre.exists() and read_json(pre)['source_plan_sha256'] == sha256(d / 'plan.json')):
            raise StageError(f'{rel(d)}/cpu_preflight is stale or partial; move it aside')
        if not pre.exists():
            self.run(f'{key}_cpu_preflight', self.rebind_argv(spec, d, preflight=True), cpu=True)
        result = read_json(pre)
        if result['status'] != 'PASS':
            raise StageError(f'{key} CPU preflight did not pass')
        self.stage(key, cpu_preflight={k: result[k] for k in ('status', 'student_depth', 'teacher_depth', 'student_parameters', 'actual_seconds')})
        if self.sim_excluded:
            log = self.run(f'{key}_sim_rebind_check', self.rebind_argv(spec, d, check='sim-cases'), cpu=True)
            got = self.last_json(log)['cases']
            if got != [n for n, _ in self.inv['sim']]:
                raise StageError('rebound simulation list differs from the inventory')
            self.stage(key, sim_rebinding={'cases': len(got), 'dropped': len(self.sim_excluded)})

    def predicted_states(self, key, d):
        target = d / 'preflight_states.json'
        if target.exists() and read_json(target).get('plan_sha256') != sha256(d / 'plan.json'):
            raise StageError(f'{rel(target)} belongs to another plan; move it aside')
        if not target.exists():
            self.run(f'{key}_states', [self.py, DRIVER / 'preflight_13_15.py', 'states', '--plan', d / 'plan.json', '--out', target], cpu=True)
            value = read_json(target)
            value['plan_sha256'] = sha256(d / 'plan.json')
            write_json(target, value)
        return read_json(target)

    def replay(self, key, spec, d):
        target = d / 'preflight_streams.json'
        if target.exists() and read_json(target).get('plan_sha256') != sha256(d / 'plan.json'):
            raise StageError(f'{rel(target)} belongs to another plan; move it aside')
        if not target.exists():
            self.run(f'{key}_replay', [self.py, DRIVER / 'preflight_13_15.py', 'replay', '--plan', d / 'plan.json', '--inventory',
                                       self.out / 'inputs' / 'inventory.json', '--sim-module', spec['sim_module'], '--out', target], cpu=True)
            value = read_json(target)
            value['plan_sha256'] = sha256(d / 'plan.json')
            write_json(target, value)
        return read_json(target)

    # ---------------------------------------------------------------- stages 13-15
    def run_stage(self, key, prev):
        spec = self.spec(key)
        d = self.out / spec['dir']
        if self.sealed(key):
            if read_json(d / 'plan.json') != self.make_plan(key, spec, prev):
                raise StageError(f'{key} is sealed but its plan differs from the plan regenerated from the current inputs '
                                 '(previous stage, stage-12 directory or code changed?)')
            self.record_outputs(key, spec, d)
            return self.prev_from_stage(key)
        if prev.get('seal') is not None:
            self.verify_seal(prev['seal'])
        plan = self.make_plan(key, spec, prev)
        t = d / 'training_candidate'
        if d.exists() and t.exists() and not (t / 'report.json').exists():
            raise StageError(f'{rel(t)} is a partial training run; move it aside, then --resume')
        d.mkdir(exist_ok=True)
        self.write_plan(d, 'plan.json', plan)
        script_spec = spec['script'] + '\n' + spec['sim_module'] + '\n'      # historical training_script.txt format
        if (d / 'training_script.txt').exists() and (d / 'training_script.txt').read_text() != script_spec:
            raise StageError(f'{rel(d)}/training_script.txt differs')
        (d / 'training_script.txt').write_text(script_spec)
        trained = (t / 'report.json').exists()
        # The trainers load <teacher_source>/training_candidate/resume_<finetune_steps>.pt only AFTER the last step
        # (train_simat.py:207): check it now, not after hours of GPU time.
        teacher_resume = prev['dir'] / 'training_candidate' / f"resume_{plan['training']['finetune_steps']}.pt"
        if not teacher_resume.exists() or (prev.get('seal') is not None and rel(teacher_resume) not in read_json(prev['seal'])['outputs']):
            if not self.smoke:
                raise StageError(f'{key}: the teacher checkpoint the trainer reads at its end is missing or unsealed: {rel(teacher_resume)}')
            self.stage(key, teacher_resume_missing_in_stand_in=rel(teacher_resume))
        if not trained:
            self.cpu_checks(key, spec, d)
        states = self.predicted_states(key, d)
        streams = self.replay(key, spec, d)
        tr = plan['training']
        self.stage(key, candidate=spec['candidate'], script=spec['script'], sim_module=spec['sim_module'], plan=rel(d / 'plan.json'),
                   plan_sha256=sha256(d / 'plan.json'), teacher=prev['candidate'], teacher_stand_in=prev['stand_in'],
                   steps=tr['steps'], warmup_steps=tr['warmup_steps'], finetune_steps=tr['finetune_steps'], middle_step=plan['middle_step'],
                   stream_seed_offset=tr['stream_seed_offset'], training=tr,
                   predicted_states={k: states[k] for k in ('teacher_point_hash', 'initial_states')},
                   student_depth=states['student_model_config']['depth'], kept_blocks=states['kept_blocks'],
                   predicted_streams=flatten_streams(streams))
        self.compare_states(key, spec, prev, states)
        self.compare_streams(key, spec, streams, d)
        kind = STAGES_13_15[key]['kind']
        if self.smoke:
            self.compare_plan(key, spec, plan, prev)
            self.gpu_step(key, kind, tr['steps'], self.rebind_argv(spec, d))
            if self.history and spec['hist'] in HIST_13_15:
                self.parity(key, spec, d / 'plan.json')
            self.stage(key, status='PLAN_GENERATED_CPU_CHECKS_OK__TRAIN_DRY_RUN')
            return self.standin_prev(key, spec, d, prev)
        if not trained:
            est = estimate_seconds(kind, tr['steps'], key)
            self.log(f'GPU {key}: ~{est[0]} s unshared on one A5500 (historical wall {est[1]} s shared)')
            self.gpu_launch(key, self.rebind_argv(spec, d))
        self.check_training(key, plan, d, states, streams)
        return self.package(key, spec, d)

    def check_training(self, key, plan, d, states, streams):
        t = d / 'training_candidate'
        report, job = read_json(t / 'report.json'), read_json(t / 'job.json')
        if report['status'] != 'COMPLETE' or report['steps_actual'] != plan['training']['steps']:
            raise StageError(f'{key} training report not COMPLETE')
        mismatch = []
        if [p['phase'] for p in report['phases']] != [p['phase'] for p in streams['phases']]:
            mismatch.append('phases')
        for got, want in zip(report['phases'], streams['phases']):
            mismatch += [f"{want['phase']}.{k}" for k in PHASE_KEYS if got[k] != want[k]]
        sim, want_sim = report.get('simulation_replay') or {}, streams.get('simulation_replay') or {}
        mismatch += [k for k in ('digest', 'samples', 'unique_windows', 'windows', 'calls') if sim.get(k) != want_sim.get(k)]
        if mismatch:
            raise StageError(f'{key} training consumed other streams than the CPU replay predicted: {mismatch}')
        got = {'teacher_point_hash': job['teacher_point_hash'], 'initial_states': job['initial_states']}
        if got != {k: states[k] for k in ('teacher_point_hash', 'initial_states')}:
            raise StageError(f'{key} training started from other states than predicted on CPU: {got}')
        if job['kept_blocks'] != states['kept_blocks'] or report['student_depth'] != states['student_model_config']['depth']:
            raise StageError(f'{key} student surgery differs from the CPU prediction')
        self.stage(key, training_checked={'streams_equal_cpu_replay': True, 'initial_states_equal_cpu_prediction': True,
                                          'training_loop_seconds': report.get('training_loop_seconds'), 'budget': report.get('budget')})

    def package(self, key, spec, d):
        audit_path = d / 'package_audit.json'
        if not audit_path.exists():
            if (d / 'runtime').exists():
                raise StageError(f'{rel(d)}/runtime exists without package_audit.json; move it aside, then --resume')
            self.run(f'{key}_package', [self.py, DRIVER / 'package_13_15.py', 'stage', '--plan', d / 'plan.json', '--out', d], cpu=True)
        audit = read_json(audit_path)
        t = d / 'training_candidate'
        paths = [d / 'plan.json', d / 'training_script.txt', t / 'report.json', t / 'job.json', *sorted(t.glob('resume_*.pt')),
                 audit_path, ROOT / audit['archive'], ROOT / audit['deployed_checkpoint'],
                 *[p for p in (d / 'runtime').rglob('*') if p.is_file()]]
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
        ref = HISTORICAL_13_15.get(spec['hist']) if self.full_release() and self.seed == SEED else None
        if ref:     # equal only if every earlier stage and this training were bit-exact
            self.report['historical_comparison'][f'{key}_deployed_states'] = compare(audit['deployed_states'], ref['deployed_states'])

    def prev_from_stage(self, key):
        spec = self.spec(key)
        d = self.out / spec['dir']
        audit = read_json(d / 'package_audit.json')
        return {'key': key, 'candidate': audit['candidate'], 'dir': d, 'zip': ROOT / audit['archive'], 'zip_sha256': audit['archive_sha256'],
                'runtime': ROOT / audit['runtime'], 'stand_in': False, 'historical': False, 'seal': self.out / 'seals' / (key + '.json')}

    def standin_prev(self, key, spec, d, prev):
        """Smoke only: the historical output of this stage (read-only) as the next stage's input. With the historical
        lineage (full release, recipe seed, default S15 options): teacher_source = the historical run dir. Otherwise: the
        smoke's own stage dir (its generated plan.json) with the historical full-release package as weight stand-in."""
        h = HIST_13_15.get(spec['hist'] or key) or HIST_13_15[key]
        archive = ROOT / h['zip']
        if not archive.exists():
            return dict(prev, key=key, candidate=spec['candidate'], dir=d, stand_in=True, historical=False, seal=None)
        historical = bool(self.history and spec['hist'] in HIST_13_15)
        return {'key': key, 'candidate': spec['candidate'], 'dir': ROOT / h['dir'] if historical else d, 'zip': archive,
                'zip_sha256': sha256(archive), 'runtime': self.extract(key, archive), 'stand_in': True, 'historical': historical,
                'seal': None}

    # ---------------------------------------------------------------- final packaging (F5 + MA) and runtime smoke
    def final(self, prev):
        cand = prev['candidate']
        d = self.out / f'final_{cand.lower()}f5ma'
        if self.sealed('final'):
            audit = read_json(d / 'final_audit.json')
            if audit['source_archive_sha256'] != prev['zip_sha256']:
                raise StageError('the sealed final package was built from another stage-15 package')
            self.stage('final', candidate=audit['candidate'], archive=audit['archive'], archive_sha256=audit['archive_sha256'])
            return ROOT / audit['runtime']
        if not self.smoke:
            self.verify_seal(prev['seal'])
        if not (d / 'final_audit.json').exists():
            if d.exists():
                raise StageError(f'{rel(d)} exists without final_audit.json; move it aside, then --resume')
            self.run('final_package', [self.py, DRIVER / 'package_13_15.py', 'final', '--source-zip', prev['zip'], '--candidate', cand,
                                       '--out', d], cpu=True)
        audit = read_json(d / 'final_audit.json')
        if audit['point_depth'] == 4 and audit['code_members_equal_ready_zip'] is not True:
            raise StageError('final runtime code members differ from the READY ZIP code members (code must be identical)')
        smoke = self.runtime_smoke('final', ROOT / audit['runtime'], d / 'runtime_smoke.json')
        self.stage('final', status='COMPLETE' if not self.smoke else 'PACKAGED_FROM_STAND_IN', candidate=audit['candidate'],
                   source=rel(prev['zip']), archive=audit['archive'], archive_sha256=audit['archive_sha256'],
                   f5_archive=audit['f5_archive'], members=len(audit['members']), point_depth=audit['point_depth'],
                   code_members_equal_ready_zip=audit['code_members_equal_ready_zip'], deployed_states=audit['deployed_states'],
                   runtime_smoke={k: smoke[k] for k in ('status', 'point_depth', 'local_blocks', 'internal_batch_size', 'checks')})
        ref = HISTORICAL_13_15.get('s15') if (self.full_release() and self.seed == SEED and cand == 'SIMSAT2D4FH') else None
        if ref:
            self.report['historical_comparison']['final_deployed_states'] = compare(audit['deployed_states'], ref['deployed_states'])
        if not self.smoke:
            paths = [d / 'final_audit.json', d / 'runtime_smoke.json', ROOT / audit['archive'], ROOT / audit['f5_archive'],
                     *[p for p in (ROOT / audit['runtime']).rglob('*') if p.is_file()]]
            self.seal('final', paths)
        self.log(f'FINAL {audit["candidate"]}: {audit["archive"]} sha256 {audit["archive_sha256"][:16]} members {len(audit["members"])} '
                 f'depth {audit["point_depth"]} code==READY {audit["code_members_equal_ready_zip"]} runtime smoke {smoke["status"]}')
        return ROOT / audit['runtime']

    def runtime_smoke(self, name, runtime, target):
        inp = self.out / 'inputs' / 'runtime_smoke_input.npy'
        if not inp.exists():
            self.run('runtime_smoke_input', [self.py, DRIVER / 'preflight_13_15.py', 'runtime-input', '--cases-json',
                                              self.out / 'inputs' / 'train_cases.json', '--out', inp], cpu=True)
        if not target.exists():
            self.run(f'{name}_runtime_smoke', [self.py, DRIVER / 'runtime_smoke_13_15.py', '--runtime', runtime, '--input', inp,
                                               '--out', target], cpu=True)
        result = read_json(target)
        if result['status'] != 'PASS':
            raise StageError(f'{name} CPU runtime smoke failed: {result["checks"]}')
        return result

    # ---------------------------------------------------------------- comparisons (smoke asserts; rebuild records)
    def reference_plan(self, key, spec):
        """(reference plan path, label, expected training changes relative to it)."""
        changes = {}
        if key == 's15':
            if self.loss_mode == 'old':
                changes.update(HIST_S15_OLD['training_changes'])
            if self.focus[0] == 'fraction' and self.focus[1] != DEFAULT_FOCUS_FRACTION:
                changes['sim_focus_fraction'] = self.focus[1]
            if self.focus[0] == 'ratio':
                changes['sim_focus_fraction'] = MISSING
                changes['sim_focus_ratio'] = self.focus[1]
            if self.s15_offset != DEFAULT_S15_OFFSET:
                changes['stream_seed_offset'] = self.s15_offset
        if self.history:
            label = 'historical full81 plan' + (f' + S15 variant changes {sorted(changes)}' if changes else '')
            if spec['hist'] == 's15_old':
                label += f' (= recorded E11 plan {HIST_S15_OLD["plan_path_recorded"]}, sha256 {HIST_S15_OLD["plan_sha256_recorded"][:16]})'
            return HIST_13_15[key]['plan'], label, changes
        if self.fold:
            return HIST_HELD_13_15[self.fold][key] + '/plan.json', f'historical held-surface plan ({self.fold})', changes
        return None, None, changes

    def compare_plan(self, key, spec, plan, prev):
        ref_path, label, changes = self.reference_plan(key, spec)
        if ref_path is None or not (ROOT / ref_path).exists():
            self.report['plan_comparison'][key] = {'status': 'SKIPPED (no historical reference for this fold/seed)'}
            return
        ref = read_json(ROOT / ref_path)
        expected_training = {k: v for k, v in {**ref['training'], **changes}.items() if v is not MISSING}
        fields = sorted({p for script, names in ((spec['script'], ('m', 'parent')), (spec['sim_module'], ('manifest',)),
                                                 (PACKAGER, ('m', 'teacher', 'fast')))
                         for p in plan_fields_read(script, names)})
        mine_parent = read_json(prev['dir'] / 'plan.json')
        ref_parent = read_json(ROOT / ref['teacher_source'] / 'plan.json') if (ROOT / ref['teacher_source'] / 'plan.json').exists() else None
        rows, unexpected = [], []
        for path in fields:
            root, keys = path[0], path[1:]
            if root == 'parent':
                a, b = dig(mine_parent, keys), dig(ref_parent, keys)
            else:
                full = ALIASES.get(root, ('m',))[1:] + keys
                a, b = dig(plan, full), dig(ref, full)
                if full[:1] == ('training',) and len(full) == 2:
                    b = expected_training.get(full[1], MISSING)
                path = ('m',) + full
            reason = next((r for p, r in INTENDED.items() if path[:len(p)] == p), None)
            if path == ('m', 'candidate') and a != b and (changes or self.fold):
                reason = 'S15 variant name' if changes else 'held-surface runs are named <X>B_<fold>'
            if path == ('m', 'training', 'seed') and a != b and self.fold and self.seed != HELD_FOLD_SEEDS[self.fold]:
                reason = f'recipe seed {self.seed}; the held surface used {HELD_FOLD_SEEDS[self.fold]} (pass --seed to match)'
            if (path == ('m', 'reflection_source') and a != b and isinstance(a, str) and isinstance(b, str)
                    and (ROOT / b).is_file() and sha256(ROOT / a) == sha256(ROOT / b)):
                reason = f'same file content (sha256 {sha256(ROOT / a)[:16]}) at another path'
            status = 'equal' if a == b else ('intended' if reason else 'UNEXPECTED')
            if status == 'UNEXPECTED':
                unexpected.append('.'.join(path))
            rows.append({'field': '.'.join(path), 'status': status, 'reason': reason if a != b else None,
                         'value': a if status != 'equal' and not isinstance(a, dict) else None,
                         'reference': b if status != 'equal' and not isinstance(b, dict) else None})
        tdiff = sorted(k for k in set(plan['training']) | set(expected_training)
                       if plan['training'].get(k, MISSING) != expected_training.get(k, MISSING))
        sdiff = sorted(k for k in set(plan['sampling']) | set(ref['sampling']) if plan['sampling'].get(k) != ref['sampling'].get(k))
        allowed = set()
        if self.fold and self.seed != HELD_FOLD_SEEDS[self.fold]:
            allowed |= {'seed'}
        result = {'reference': ref_path, 'label': label, 'expected_training_changes': {k: (None if v is MISSING else v) for k, v in changes.items()},
                  'fields_compared': len(rows), 'equal': sum(r['status'] == 'equal' for r in rows),
                  'intended': sum(r['status'] == 'intended' for r in rows), 'unexpected': unexpected,
                  'training_keys_differing': tdiff, 'sampling_keys_differing': sdiff, 'rows': rows}
        if self.fold:           # held plans carry descriptive drift; only keys a script reads are fatal (as stages 8-12)
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
        self.log(f'PLAN {key} vs {ref_path}: {result["equal"]} equal, {result["intended"]} intended, unexpected={unexpected} '
                 f'training_diff={tdiff} sampling_diff={sdiff}')
        if result['status'] != 'OK':
            raise StageError(f'{key} plan differs from {ref_path} beyond the intended differences: {unexpected + bad}')

    def stream_refs(self, key, spec):
        """Historical stream references for this stage/options (full release + recipe seed), or ({}, reason)."""
        if not (self.full_release() and self.seed == SEED):
            return {}, None
        if key == 's15' and self.s15_offset != DEFAULT_S15_OFFSET:
            return {}, None                           # another real stream: no historical full81 reference
        refs = dict(HISTORICAL_13_15[key]['streams'])
        if key == 's15' and self.focus != ('fraction', DEFAULT_FOCUS_FRACTION):
            refs = {k: v for k, v in refs.items() if not k.startswith('sim_')}      # other focus settings: other sim stream
            return refs, 'real streams only (the simulation stream depends on the focus settings)'
        return refs, 'S15 historical streams (the loss mode does not change any stream)' if key == 's15' and self.loss_mode == 'old' else None

    def compare_streams(self, key, spec, streams, d):
        mine = flatten_streams(streams)
        refs, note = self.stream_refs(key, spec)
        extra = {}
        if not refs and self.fold and self.seed == HELD_FOLD_SEEDS[self.fold] and self.smoke:
            refs, extra = self.held_stream_refs(key, spec, d, streams)
        if not refs:
            self.report['stream_comparison'][key] = {'status': 'SKIPPED (no historical reference for this fold/seed/options)'}
            return
        rows = {k: compare(mine.get(k, MISSING), v) for k, v in refs.items()}
        self.report['stream_comparison'][key] = {'compared': len(rows), 'match': sum(v['match'] for v in rows.values()),
                                                 'differing': [k for k, v in rows.items() if not v['match']], 'note': note, **extra}
        self.verification(f'streams_{key}', {'replayed': streams, 'reference': refs, 'comparison': rows, 'note': note, 'held': extra})
        bad = [k for k, v in rows.items() if not v['match']]
        self.log(f'STREAMS {key}: {len(rows) - len(bad)}/{len(rows)} historical stream values reproduced' + (f'; DIFF {bad}' if bad else '')
                 + (f' | held sim protocol: {json.dumps(extra)}' if extra else ''))
        if bad and self.smoke:
            raise StageError(f'{key} replayed streams differ from the historical run: {bad}')

    def held_stream_refs(self, key, spec, d, streams):
        """Held-surface runs (fold smoke with the fold's seed): phase statistics from their report.json; their simulation
        replay used ALL simulations (historical protocol), which is replayed separately to validate the sampler code."""
        h = HIST_HELD_13_15[self.fold][key]
        path = ROOT / h / 'training_candidate' / 'report.json'
        if spec['hist'] != key or not path.exists():
            return {}, {}
        r = read_json(path)
        refs = {f"{ph['phase']}.{k}": ph[k] for ph in r['phases'] for k in PHASE_KEYS if k in ph}
        extra = {}
        held_sim = r.get('simulation_replay') or {}
        if held_sim and streams.get('simulation_replay'):
            inv_all = self.out / 'inputs' / 'inventory_allsims.json'
            if not inv_all.exists():
                write_json(self.out / 'inputs' / 'no_exclusion.json', [])
                self.run('inventory_allsims', [self.py, DRIVER / 'preflight_8_12.py', 'inventory', '--cases-json',
                                               self.out / 'inputs' / 'train_cases.json', '--sim-excluded-json',
                                               self.out / 'inputs' / 'no_exclusion.json', '--out', inv_all], cpu=True)
            target = d / 'preflight_streams_allsims.json'
            if not target.exists():
                self.run(f'{key}_replay_allsims', [self.py, DRIVER / 'preflight_13_15.py', 'replay', '--plan', d / 'plan.json',
                                                   '--inventory', inv_all, '--sim-module', spec['sim_module'], '--out', target], cpu=True)
            sim = read_json(target)['simulation_replay']
            extra = {'historical_protocol_all_sims_digest_match': sim['digest'] == held_sim['digest'],
                     'historical_protocol_samples': [sim['samples'], held_sim['samples']],
                     'fold_rule_digest_differs_by_design': streams['simulation_replay']['digest'] != held_sim['digest']}
            if not extra['historical_protocol_all_sims_digest_match']:
                raise StageError(f'{key}: the simulation replay of the historical held protocol does not reproduce {rel(path)}')
        return refs, extra

    def compare_states(self, key, spec, prev, states):
        if not prev.get('historical') and not (self.full_release() and self.seed == SEED and not self.smoke):
            return       # fold/seed smokes use full-release weights as plumbing stand-ins: nothing meaningful to compare
        ref = HISTORICAL_13_15[key]      # the set-up depends on the teacher and the student architecture only (not on loss/focus)
        mine = {k: states[k] for k in ('teacher_point_hash', 'initial_states')}
        want = {k: ref[k] for k in ('teacher_point_hash', 'initial_states')}
        row = compare(mine, want)
        self.report['state_comparison'][key] = {'match': row['match'], 'teacher_is_historical': prev.get('historical', False)}
        if prev.get('historical'):
            self.log(f'STATES {key}: CPU-predicted teacher/initial state hashes {"equal" if row["match"] else "DIFFER from"} the historical job.json')
            if self.smoke and not row['match']:
                raise StageError(f'{key}: the model set-up on the historical teacher does not reproduce the historical initial states')
        else:
            self.report['historical_comparison'][f'{key}_initial_states'] = row

    # ---------------------------------------------------------------- packaging parity (smoke)
    def parity(self, key, spec, plan_path):
        h = HIST_13_15[spec['hist']]
        t = ROOT / h['dir'] / 'training_candidate'
        hist_plan = ROOT / h['plan']
        variants = {'historical_plan': [hist_plan, ROOT / h['dir'] / 'training_script.txt'], 'rebuilt_plan': [plan_path, None]}
        result = {'reference_zip': h['zip'], 'reference_checkpoint': h['checkpoint'], 'variants': {}}
        for name, (plan, script) in variants.items():
            out = self.out / 'parity' / key / name
            argv = [self.py, DRIVER / 'package_13_15.py', 'stage', '--plan', plan, '--trained-plan', hist_plan, '--training-dir', t, '--out', out]
            if script:
                argv += ['--script-spec', script]
            self.run(f'parity_{key}_{name}', argv, cpu=True)
            audit = read_json(out / 'package_audit.json')
            cmp = self.compare_zip(f'package_{key}_{name}', ROOT / audit['archive'], ROOT / h['zip'])
            cmp['deployed_checkpoint_identical'] = audit['deployed_checkpoint_sha256'] == sha256(ROOT / h['checkpoint'])
            differences, unexplained = [], []
            for row in cmp['rows']:
                if row['status'] != 'identical':
                    why = 'continuation' if (name == 'rebuilt_plan' and row['member'] == f'training_recipe/{audit["candidate"]}_continuation.json') else None
                    differences.append({'member': row['member'], 'status': row['status'], 'reason': why})
                    if why is None:
                        unexplained.append(row['member'])
            if not cmp['deployed_checkpoint_identical']:
                unexplained.append('deployed checkpoint')
            result['variants'][name] = {'plan': rel(plan), 'template_archive': audit['template_archive']['path'],
                                        'members_rebuilt': cmp['members_rebuilt'], 'members_reference': cmp['members_reference'],
                                        'identical': cmp['identical'], 'differences': differences, 'unexplained': unexplained,
                                        'common_member_order_equal': cmp['common_member_order_equal'],
                                        'deployed_checkpoint_identical': cmp['deployed_checkpoint_identical'], 'archive': audit['archive']}
            self.log(f'PARITY {key}/{name}: {cmp["identical"]}/{cmp["members_reference"]} reference members byte-identical; '
                     f'differences {[(x["member"], x["reason"]) for x in differences]}; checkpoint identical={cmp["deployed_checkpoint_identical"]}')
            if unexplained or not cmp['common_member_order_equal']:
                raise StageError(f'{key}/{name} package differs from {h["zip"]} without a known reason: {unexplained}')
        result['reasons'] = PACKAGE_REASONS
        self.report['package_parity'][key] = result

    def compare_zip(self, name, rebuilt, reference):
        target = self.out / 'verification' / f'{name}.json'
        self.run(f'compare_{name}', [self.py, DRIVER / 'preflight_8_12.py', 'compare', '--rebuilt', rebuilt, '--reference', reference,
                                     '--out', target], cpu=True)
        return read_json(target)

    def final_parity(self):
        """Smoke: F5 + MA on the parity S15 package (historical plan) vs the historical F5 / F5MA ZIPs (= READY ZIP), and the
        CPU runtime smoke of the rebuilt final runtime vs the READY ZIP runtime (same predictions expected)."""
        source = self.out / 'parity' / 's15' / 'historical_plan' / 'SIMSAT2D4FH_REBUILT.zip'
        d = self.out / 'parity' / 'final'
        self.run('parity_final_package', [self.py, DRIVER / 'package_13_15.py', 'final', '--source-zip', source, '--candidate', 'SIMSAT2D4FH',
                                          '--out', d], cpu=True)
        audit = read_json(d / 'final_audit.json')
        result = {}
        for tag, mine in (('f5', audit['f5_archive']), ('f5ma', audit['archive'])):
            ref = HIST_FINAL[tag]
            cmp = self.compare_zip(f'final_{tag}', ROOT / mine, ROOT / ref['path'])
            bad = [r['member'] for r in cmp['rows'] if r['status'] != 'identical']
            result[tag] = {'reference': ref['path'], 'identical': cmp['identical'], 'members_reference': cmp['members_reference'],
                           'members_rebuilt': cmp['members_rebuilt'], 'order_equal': cmp['common_member_order_equal'], 'differing': bad}
            self.log(f'PARITY final/{tag}: {cmp["identical"]}/{cmp["members_reference"]} members of {ref["path"]} byte-identical; differing {bad}')
            if bad or not cmp['common_member_order_equal'] or cmp['members_rebuilt'] != cmp['members_reference']:
                raise StageError(f'final {tag} package differs from {ref["path"]}: {bad}')
        ready_rt = self.extract('ready', ROOT / HIST_FINAL['ready']['path'])
        a = self.runtime_smoke('parity_final', ROOT / audit['runtime'], d / 'runtime_smoke.json')
        b = self.runtime_smoke('ready_zip', ready_rt, self.out / 'standins' / 'ready' / 'runtime_smoke.json')
        result['runtime_smoke'] = {'rebuilt': a['status'], 'ready_zip': b['status'],
                                   'prediction_hashes_equal': a['prediction_sha256'] == b['prediction_sha256'],
                                   'checks': a['checks'], 'point_depth': a['point_depth']}
        result['code_members_equal_ready_zip'] = audit['code_members_equal_ready_zip']
        self.log(f'PARITY final runtime smoke (CPU): rebuilt {a["status"]}, READY {b["status"]}, predictions equal '
                 f'{result["runtime_smoke"]["prediction_hashes_equal"]}')
        if not result['runtime_smoke']['prediction_hashes_equal']:
            raise StageError('CPU predictions of the rebuilt final runtime differ from the READY ZIP runtime')
        self.report['final_parity'] = result

    # ---------------------------------------------------------------- main flow
    def execute(self):
        self.prologue()
        self.resolve_inputs()
        prev = self.s12
        for key in self.keys:
            prev = self.run_stage(key, prev)
        if self.smoke:
            if self.history and self.spec('s15')['hist'] == 's15':
                self.final_parity()
            else:     # plumbing: the historical S15 package under this variant's name
                self.final(dict(prev, zip=ROOT / HIST_13_15['s15']['zip']))
            return None
        return self.final(prev)


def summarize_smoke(driver):
    r = driver.report
    driver.log('SMOKE SUMMARY (CPU only; nothing ran on the GPU)')
    for name, values in r['stages'].items():
        driver.log(f'  stage {name}: {values.get("status", "")}')
    for key, value in r['plan_comparison'].items():
        driver.log(f'  plan {key}: {json.dumps({k: value.get(k) for k in ("status", "reference", "equal", "intended", "unexpected", "training_keys_differing")})}')
    for key, value in r['stream_comparison'].items():
        driver.log(f'  streams {key}: {json.dumps(value)}')
    for key, value in r['state_comparison'].items():
        driver.log(f'  initial states {key}: {json.dumps(value)}')
    for key, value in r['package_parity'].items():
        for name, v in value['variants'].items():
            driver.log(f'  package {key}/{name}: {v["identical"]}/{v["members_reference"]} byte-identical; differences '
                       f'{[(x["member"], x["reason"]) for x in v["differences"]]}; checkpoint identical {v["deployed_checkpoint_identical"]}')
    if r['final_parity']:
        fp = r['final_parity']
        driver.log(f'  final F5 {fp["f5"]["identical"]}/{fp["f5"]["members_reference"]}, F5MA {fp["f5ma"]["identical"]}/{fp["f5ma"]["members_reference"]} '
                   f'byte-identical to the historical/READY ZIPs; CPU runtime smoke {json.dumps(fp["runtime_smoke"])}')
    for step in r['gpu_steps']:
        e = step['estimate_seconds']
        driver.log(f'  GPU needed: {step["stage"]} ({step["kind"]}, {step["steps"]} updates) ~{e["unshared"]} s unshared; '
                   f'historical {e["historical_logged_shared"]} s shared')
    if r.get('gpu_estimate_total_seconds'):
        t = r['gpu_estimate_total_seconds']
        driver.log(f'  GPU total (one A5500): ~{t["unshared"]} s unshared; historical {t["historical_logged_shared"]} s shared')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='new directory under runs/FINAL_REBUILD/')
    parser.add_argument('--stage12', help='COMPLETE rebuild_8_12.py output directory (required unless --smoke)')
    parser.add_argument('--loss-mode', choices=['default', 'old'], default='default',
                        help="S15 real loss: default = C983 composite (train_simati_focus.py); old = pre-C983 (train_simati_focus_old.py, E11)")
    parser.add_argument('--focus-fraction', type=float, help=f'S15 sim_focus_fraction (default {DEFAULT_FOCUS_FRACTION}; round(8F) focus windows per batch)')
    parser.add_argument('--focus-ratio', type=float, help='S15 sim_focus_ratio (focus2 sampler: fixed per-window emphasis, accumulator)')
    parser.add_argument('--s15-offset', type=int, default=DEFAULT_S15_OFFSET,
                        help=f'S15 real-data stream_seed_offset (default {DEFAULT_S15_OFFSET}; the E9 focus-ratio run used 2100)')
    parser.add_argument('--holdout-re', help="smoke without --stage12: comma list of Reynolds numbers and/or one '>=X' term")
    parser.add_argument('--holdout-aoa', help='smoke without --stage12: comma list of angles of attack')
    parser.add_argument('--seed', type=int, help=f'recipe seed (inherited from --stage12; smoke default {SEED})')
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
