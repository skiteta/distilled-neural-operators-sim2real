"""FINAL_REBUILD driver, stages 1-7: random init -> C787H-equivalent runtime/ (the LATCADF input).

Spec: runs/FINAL_REBUILD/spec_stages_1_to_7.md (section 9). Only new code lives here; the
historical training code runs unchanged:
  S1-S4  C675 weight-free pipeline `python -m realpde.c628_pipeline prepare/execute`   (GPU)
  S5     stage5_c728p.c728p_convert                                                    (CPU)
  S6     runs/C758R_full81_exposure_matched/train.py via rebind_train.py, generated plan (GPU)
  S7     runs/C787R_full81_point_preserved/train.py via rebind_train.py, generated plan (GPU)
         + c787h_package.c787h_package -> OUT/s7_c787h/runtime/                        (CPU)
Code and data pins are asserted. Weight/checkpoint/ZIP hashes are recorded, never asserted;
rebuild_report.json lists them with the historical references as {value, historical, match}.

Usage (any cwd; a relative --out is taken relative to the project root):
  .venv310/bin/python runs/FINAL_REBUILD/driver/rebuild_1_7.py --out runs/FINAL_REBUILD/<RID>
      [--holdout-re LIST] [--holdout-aoa LIST] [--smoke] [--resume]
  --holdout-re  comma list of exact Reynolds numbers and/or one '>=X' term, e.g. '8850,10125' or '>=19050'
  --holdout-aoa comma list of angles, e.g. '15' or '5,15'
  Held-out regimes are removed from sim AND real training of every stage; the case lists are
  written to OUT/holdout.json. --smoke is CPU-only: it prepares/validates every plan, runs the
  CPU stages, runs parity checks against the historical lineage and dry-runs the GPU stages.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C253_CONFIG, C253_SCRIPT, C757_TRAINING, CLOSURE, CODE_PINS, COMPONENT_FILES,  # noqa: E402
                    DATA, DRIVER, ENV, EXCLUDED_REAL, GPU_ESTIMATES, HEAD_FILES, HIST_C628_PARENT, HIST_C728_DIR,
                    HIST_C758_TRAINING, HIST_C787_DIR, HIST_C787H_ZIP, HISTORICAL, REBUILD_BASE, REFERENCE_STEPS,
                    REFERENCE_WINDOWS, RETAINED_REAL, ROOT, RUNTIME_CODE, RUNTIME_CODE_DIR, S7_ENDPOINT_FRACTION,
                    T6, T7, compare, read_json, regime, rel, release_names, sha256, write_json)
import c675_ext  # noqa: E402  (imports realpde.c628_pipeline; stdlib only, no torch in the driver process)

C675_MAX_WALL = 43200
C675_MAX_GIB = 3
# Smoke-only, read-only parity references (absent in a weight-free shipped tree -> parity skipped).
HIST_ARTIFACTS = [HIST_C628_PARENT, HIST_C728_DIR + '/initialization_report.json', HIST_C728_DIR + '/common_runtime',
                  HIST_C758_TRAINING + '/resume_9232.pt', HIST_C758_TRAINING + '/job.json',
                  HIST_C787_DIR + '/candidate_plan.json', HIST_C787_DIR + '/package_audit.json', HIST_C787H_ZIP]
DRIVER_FILES = ['common.py', 'rebuild_1_7.py', 'stage5_c728p.py', 'c787h_package.py', 'rebind_train.py',
                'preflight.py', 'c675_ext.py']


class StageError(RuntimeError):
    pass


def parse_re(text):
    values, ge = set(), None
    for token in filter(None, (t.strip() for t in (text or '').split(','))):
        if token.startswith('>='):
            if ge is not None:
                raise SystemExit('--holdout-re accepts one >= term')
            ge = int(token[2:])
        else:
            values.add(int(token))
    return sorted(values), ge


def parse_aoa(text):
    return sorted({int(t) for t in (text or '').split(',') if t.strip()})


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
        self.out, self.args, self.smoke = out, args, args.smoke
        self.owned = False                    # only a directory this run created/resumed is ever written or cleaned
        self.py = args.python or sys.executable
        self.re_values, self.re_ge = parse_re(args.holdout_re)
        self.aoa_values = parse_aoa(args.holdout_aoa)
        names = release_names('train_sim') + release_names('train_real')
        unknown = ([v for v in self.re_values if v not in {regime(n)[0] for n in names}]
                   + [v for v in self.aoa_values if v not in {regime(n)[1] for n in names}])
        if unknown:
            raise SystemExit(f'holdout values not in the release: {unknown}')
        self.holdout = None
        self.history = False                  # smoke: historical artifacts present for parity checks
        self.report = {'status': 'RUNNING', 'mode': 'smoke_cpu' if self.smoke else 'rebuild', 'out': None,
                       'stages': {}, 'historical_comparison': {}, 'gpu_steps': [], 'commands': []}
        self.started = time.monotonic()

    # ---------------------------------------------------------------- utilities
    def log(self, message):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
        print(line, flush=True)
        if self.owned:
            with (self.out / 'driver.log').open('a') as f:
                f.write(line + '\n')

    def env(self, extra=None, cpu=False):
        env = dict(os.environ)
        env.update(ENV)
        env.pop('PYTHONPATH', None)          # every entry point sets its own path (run_stage.py:15 did the same)
        env.update(extra or {})
        if cpu or self.smoke:
            env['CUDA_VISIBLE_DEVICES'] = ''
        return env

    def run(self, name, argv, extra=None, cpu=False):
        argv = [str(a) for a in argv]
        log = self.out / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        self.log(f'RUN {name}' + (' [CPU]' if cpu or self.smoke else ' [GPU]'))
        with log.open('ab') as stream:
            stream.write(('\n$ ' + ' '.join(argv) + '\n').encode())
            stream.flush()
            proc = subprocess.run(argv, cwd=ROOT, env=self.env(extra, cpu), stdout=stream, stderr=subprocess.STDOUT)
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

    def gpu_step(self, stage, argv):
        estimate = GPU_ESTIMATES[stage]
        self.report['gpu_steps'].append({'stage': stage, 'command': [str(a) for a in argv],
                                         'estimate_seconds': estimate['seconds'], 'note': estimate['note']})
        self.log(f'DRY-RUN {stage}: GPU step not run (est. {estimate["seconds"][0]}-{estimate["seconds"][1]} s)')

    def seal(self, stage, paths):
        write_json(self.out / 'seals' / (stage + '.json'), {'stage': stage, 'outputs': {rel(p): sha256(p) for p in paths}})

    def sealed(self, stage):
        path = self.out / 'seals' / (stage + '.json')
        if not path.exists():
            return False
        bad = [p for p, h in read_json(path)['outputs'].items() if sha256(ROOT / p) != h]
        if bad:
            raise StageError(f'sealed outputs of {stage} changed: {bad[:3]}')
        self.log(f'REUSE {stage} (seal verified)')
        self.stage(stage, status='REUSED_SEALED')
        return True

    def hashes(self, name, paths):
        """sha256 + state hash of checkpoint files, computed in a CPU child (records only)."""
        spec, target = self.out / 'inputs' / (name + '_paths.json'), self.out / 'inputs' / (name + '_hashes.json')
        write_json(spec, [rel(p) for p in paths])
        target.unlink(missing_ok=True)
        self.run('hash_' + name, [self.py, DRIVER / 'preflight.py', 'hashes', '--paths-json', spec, '--out', target], cpu=True)
        return read_json(target)

    def full_release(self):
        return self.holdout['mode'] == 'none'

    # ---------------------------------------------------------------- prologue
    def prologue(self):
        manifest = {'holdout_re': self.re_values, 'holdout_re_ge': self.re_ge, 'holdout_aoa': self.aoa_values,
                    'smoke': self.smoke, 'python': self.py}
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
        self.log(f'START {self.report["mode"]} out={rel(self.out)} python={self.py}')
        for path, expected in CODE_PINS.items():                                   # code pins (asserted)
            if sha256(ROOT / path) != expected:
                raise StageError('code pin mismatch: ' + path)
        vendored = {str(p.relative_to(RUNTIME_CODE_DIR)): sha256(p) for p in sorted(RUNTIME_CODE_DIR.rglob('*')) if p.is_file()}
        if vendored != RUNTIME_CODE:
            raise StageError('vendored runtime code differs from its pins')
        names = release_names('train_real')                                        # data pins (asserted)
        assert len(names) == 81 and RETAINED_REAL in names and EXCLUDED_REAL not in names
        # Decision-phase data omits train_real/7575_0.h5; it was always excluded from training (EXCLUDED_REAL),
        # so its presence is no longer asserted.
        assert len(release_names('train_sim')) == 100
        write_json(self.out / 'cases81.json', names)
        if not (self.out / 'environment.json').exists():
            self.environment()

    def environment(self):
        probe = ("import json,platform,sys,torch,numpy,h5py,yaml;print(json.dumps({'python':sys.version.split()[0],"
                 "'executable':sys.executable,'venv':sys.prefix!=sys.base_prefix,'torch':torch.__version__,"
                 "'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,"
                 "'h5py':h5py.__version__,'yaml':yaml.__version__,'platform':platform.platform()}))")
        log = self.run('environment_probe', [self.py, '-c', probe], cpu=True)
        env = {'software': self.last_json(log), 'env': ENV}
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=20)
            env['gpu'] = gpu.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            env['gpu'] = repr(error)
        files = CLOSURE + [str((DRIVER / f).relative_to(ROOT)) for f in DRIVER_FILES]
        files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'submission/c628_runtime').rglob('*') if p.is_file())
        env['code_sha256'] = {f: sha256(ROOT / f) for f in files}
        env['vendored_runtime_code'] = {'dir': rel(RUNTIME_CODE_DIR), 'sha256': RUNTIME_CODE}
        write_json(self.out / 'environment.json', env)

    # ---------------------------------------------------------------- holdout
    def resolve_holdout(self):
        sim, real = release_names('train_sim'), release_names('train_real')
        all_re = {regime(n)[0] for n in sim + real}
        requested = {'holdout_re_values': self.re_values, 'holdout_re_ge': self.re_ge, 'holdout_aoa_values': self.aoa_values}
        native, ext = {}, None
        if not self.re_values and self.re_ge is None and not self.aoa_values:
            mode, parts = 'none', c675_ext.STOCK_SPLIT(sim, real)
        else:
            parts = c675_ext.split_inventory(sim, real, holdout_re_ge=self.re_ge, holdout_re_values=self.re_values,
                                             holdout_aoa_values=self.aoa_values)
            native = None
            if len(self.aoa_values) <= 1:     # stock flags: one Re threshold and one AoA; use them if identical
                held_re = sorted(set(self.re_values) | {r for r in all_re if self.re_ge is not None and r >= self.re_ge})
                candidate = {'holdout_re_ge': held_re[0] if held_re else None,
                             'holdout_aoa': self.aoa_values[0] if self.aoa_values else None}
                try:
                    native = candidate if c675_ext.STOCK_SPLIT(sim, real, **candidate) == parts else None
                except ValueError:
                    native = None
            mode = 'c675_native' if native is not None else 'c675_list_extension'
            if native is None:
                ext = {'holdout_re_ge': self.re_ge, 'holdout_aoa': None, 'holdout_re_values': self.re_values,
                       'holdout_aoa_values': self.aoa_values}
        holdout = {'mode': mode, 'requested': requested, 'c675_native_args': native, 'c675_list_holdout': ext,
                   'counts': {k: len(v) for k, v in parts.items()},
                   'cases': {k: parts[k] for k in ('sim_train', 'sim_excluded', 'real_train', 'real_evaluation')},
                   'inner_folds_held_reynolds': [f['held_reynolds'] for f in parts['inner_folds']],
                   'propagation': 'real_train is the training case list of S1-S7; real_evaluation and sim_excluded '
                                  'are never loaded for training (S5-S7 read only real_train arrays; h5 metadata of '
                                  'all release files is inspected by discover_cases, as in C675).'}
        if (self.out / 'holdout.json').exists():
            if read_json(self.out / 'holdout.json') != holdout:
                raise StageError('holdout.json differs from the recomputed partition')
        else:
            write_json(self.out / 'holdout.json', holdout)
            write_json(self.out / 'train_cases.json', parts['real_train'])
        self.holdout, self.parts = holdout, parts
        self.log(f'HOLDOUT mode={mode} counts={holdout["counts"]}')
        self.stage('holdout', mode=mode, counts=holdout['counts'], c675_native_args=native, file=rel(self.out / 'holdout.json'))

    # ---------------------------------------------------------------- S1-S4 (C675)
    def stage1_4(self):
        d = self.out / 's1_4_c675'
        plan_path = d / 'plan.json'
        pythonpath = {'PYTHONPATH': f'{ROOT / "src"}{os.pathsep}{ROOT}'}
        ext = DRIVER / 'c675_ext.py'
        if self.holdout['mode'] == 'c675_list_extension':
            prep = [self.py, ext, 'prepare', '--data-root', DATA, '--output', d,
                    '--holdout-json', json.dumps(self.holdout['c675_list_holdout'])]
            exe = [self.py, ext, 'execute']
        else:
            prep = [self.py, '-m', 'realpde.c628_pipeline', 'prepare', '--data-root', DATA, '--output', d]
            for key, flag in (('holdout_re_ge', '--holdout-re-ge'), ('holdout_aoa', '--holdout-aoa')):
                if (self.holdout['c675_native_args'] or {}).get(key) is not None:
                    prep += [flag, self.holdout['c675_native_args'][key]]
            exe = [self.py, '-m', 'realpde.c628_pipeline', 'execute']
        exe += ['--plan', plan_path, '--max-wall-seconds', self.args.c675_max_wall_seconds,
                '--max-output-gib', self.args.c675_max_output_gib]
        if not self.sealed('s1_4_c675'):
            if not plan_path.exists():
                if d.exists():
                    raise StageError(f'{rel(d)} exists without plan.json; inspect and move it aside')
                self.run('s1_4_c675_prepare', prep, pythonpath, cpu=True)
            plan = read_json(plan_path)
            if ({k: plan['cases'][k] for k in self.holdout['cases']} != self.holdout['cases']
                    or plan['cases']['inner_folds'] != self.parts['inner_folds']):
                raise StageError('C675 plan partition differs from the driver partition')
            self.run('s1_4_c675_check', [self.py, ext, 'check', '--plan', plan_path], pythonpath, cpu=True)
            self.stage('s1_4_c675', plan=rel(plan_path), plan_sha256=sha256(plan_path), holdout=plan['holdout'],
                       cases={k: len(v) for k, v in plan['cases'].items()}, execute_command=[str(a) for a in exe])
            if self.smoke:
                self.gpu_step('s1_4_c675', exe)
                self.stage('s1_4_c675', status='PREPARED_AND_VALIDATED__EXECUTE_DRY_RUN')
                return None
            self.run('s1_4_c675_execute', exe, pythonpath)
            last = read_json(sorted(d.glob('attempt_*.json'))[-1])
            if last.get('status') != 'TRAINING_AND_EXPORT_COMPLETE_EVALUATION_REQUIRED':
                raise StageError('C675 execute did not complete: ' + str(last.get('status')))
            for node in plan['nodes']:
                assert read_json(d / node['id'] / 'complete.json')['outer_targets_used_for_training'] is False
            for node in ('joint', 'sps', 'hidden', 'observed'):
                assert read_json(d / node / 'report.json')['status'] == 'COMPLETE', node
            self.seal('s1_4_c675', [plan_path, *(d / s for s in plan['components'].values()), d / 'joint/final_head.pt'])
        plan = read_json(plan_path)
        components = {name: d / source for name, source in plan['components'].items()}
        hashed = self.hashes('s1_4', [*components.values(), d / 'joint/final_head.pt'])
        states = {name: hashed[rel(path)] for name, path in components.items()}
        self.stage('s1_4_c675', status='COMPLETE', components={n: {'path': rel(components[n]), **states[n]} for n in components},
                   joint_head=hashed[rel(d / 'joint/final_head.pt')])
        if self.full_release():
            hc = self.report['historical_comparison']
            for key, name in (('s1_point', 'model.pt'), ('s2_center', 'center_head.pt'), ('s2_radius', 'radius_head.pt'),
                              ('s3_hidden', 'hidden_head.pt'), ('s4_observed', 'observed_center_head.pt')):
                hc[key] = compare(states[name]['state_hash'], HISTORICAL[key])
            hc['s1_head'] = compare(hashed[rel(d / 'joint/final_head.pt')]['state_hash'], HISTORICAL['s1_head'])
        return components

    def smoke_stage1_cpu(self):
        """Historical C253 script in its own --smoke mode (1 step per sub-stage) on CPU.

        The C675 execute path cannot be shrunk (validate() pins the fixed recipe config) and
        S2-S4 hard-code torch.device('cuda'); the C253 script is the stage-1 entry point that
        supports a tiny CPU run. It uses the full release inventory (no holdout support)."""
        d = self.out / 's1_c253_cpu_smoke'
        self.run('s1_c253_cpu_smoke', [self.py, C253_SCRIPT, '--config', C253_CONFIG, '--output-dir', d,
                                       '--smoke', '--smoke-steps', '1', '--smoke-cache-batches', '1'], cpu=True)
        report = read_json(d / 'report.json')
        if report['status'] != 'SMOKE_PASS':
            raise StageError('C253 CPU smoke status ' + report['status'])
        self.stage('s1_c253_cpu_smoke', status='SMOKE_PASS (historical S1 code, 1 step per sub-stage, CPU)',
                   final_point_state=report['learned_state_hashes']['final_point'],
                   note='ignores --holdout (the C675 path honors it)')
        return d / 'final_point.pt'

    # ---------------------------------------------------------------- S5
    def stage5(self, components, name='s5_c728p', cases_file=None):
        out5 = self.out / name
        if not self.sealed(name):
            if out5.exists():
                raise StageError(f'{rel(out5)} exists without a seal; inspect and move it aside')
            parent = self.out / 'inputs' / (name + '_parent.json')
            write_json(parent, {n: rel(p) for n, p in components.items()})
            self.run(name, [self.py, DRIVER / 'stage5_c728p.py', '--parent-json', parent,
                            '--cases', cases_file or self.out / 'train_cases.json', '--out', out5], cpu=True)
            self.seal(name, [p for p in out5.rglob('*') if p.is_file()])
        init = read_json(out5 / 'initialization_report.json')
        self.stage(name, status='COMPLETE (CPU)', initial_states=init['initial_states'], train_cases=init['train_cases'],
                   windows=init['train_windows'], identity_checks=len(init['CPU_identity_checks']),
                   model_sha256=init['runtime_members']['model.pt'],
                   parent={k: v['path'] for k, v in init['parent_components'].items()})
        if name == 's5_c728p' and self.full_release() and not self.smoke:
            self.report['historical_comparison']['s5_point'] = compare(init['initial_states']['point'], HISTORICAL['s5_point'])
            self.report['historical_comparison']['s5_windows'] = compare(init['train_windows'], HISTORICAL['s5_windows'])
        return out5 / 'common_runtime'

    # ---------------------------------------------------------------- S6
    def s6_plan(self, rt5):
        out5 = rt5.parent
        init, dp = read_json(out5 / 'initialization_report.json'), read_json(out5 / 'dense_plan.json')
        for member, digest in init['runtime_members'].items():                     # chain self-check
            if sha256(rt5 / member) != digest:
                raise StageError('S5 runtime changed: ' + member)
        if dp['train_cases'] != self.parts['real_train']:
            raise StageError('S5 case list differs from the holdout partition')
        windows = dp['sampling']['windows']
        steps = math.ceil(REFERENCE_STEPS * windows / REFERENCE_WINDOWS)            # C758R prepare.py:10
        training = dict(C757_TRAINING)
        training.update(steps=steps, schedule_steps=2 * steps, EMA=.999 ** (6000 / steps))
        pins = {rel(rt5 / m): h for m, h in init['runtime_members'].items()}
        pins[T6] = sha256(ROOT / T6)
        return {'candidate': 'C758H', 'runtime': rel(rt5), 'source_pins': pins, 'train_cases': dp['train_cases'],
                'eval_cases': [], 'training': training, 'sampling': dp['sampling'], 'initial_states': init['initial_states'],
                'arms': {'ema': {'candidate': 'C758H'}, 'live': {'candidate': 'C758H_LIVE_NOT_SELECTED'}},
                'holdout': {'mode': self.holdout['mode'], 'real_evaluation': self.parts['real_evaluation'],
                            'sim_excluded': len(self.parts['sim_excluded'])},
                'generated_by': rel(DRIVER / 'rebuild_1_7.py'),
                'policy': 'C757R data-exposure rule inlined: steps=ceil(6000*windows/42847), schedule=2*steps, EMA=.999**(6000/steps)'}

    def rebind(self, script, stage_dir, check=None):
        argv = [self.py, DRIVER / 'rebind_train.py', '--script', script, '--sha256', CODE_PINS[script],
                '--stage-dir', stage_dir, '--arm', 'candidate', '--seconds', '0']
        if script == T6 and len(self.parts['real_train']) != 81:
            argv += ['--train-case-count', len(self.parts['real_train'])]
        return argv + (['--check', check] if check else [])

    def stage6(self, rt5):
        d = self.out / 's6_c758h'
        t = d / 'training_candidate'
        if self.sealed('s6_c758h'):
            return d
        plan = self.s6_plan(rt5)
        if d.exists():
            if not (d / 'candidate_plan.json').exists() or read_json(d / 'candidate_plan.json') != plan:
                raise StageError(f'{rel(d)} holds a different plan; inspect and move it aside')
            if t.exists() and not (t / 'report.json').exists():
                raise StageError(f'{rel(t)} is a partial training run; move it aside, then --resume')
        else:
            d.mkdir()
            write_json(d / 'candidate_plan.json', plan)
        trained = (t / 'report.json').exists()
        if not trained:
            check = self.run('s6_rebind_check', self.rebind(T6, d, 'plan'), cpu=True)
            self.run('s6_preflight', [self.py, DRIVER / 'preflight.py', 's6', '--stage-dir', d], cpu=True)
            self.stage('s6_c758h', rebind=self.last_json(check))
        pf = read_json(d / 'preflight.json')
        self.stage('s6_c758h', plan=rel(d / 'candidate_plan.json'), plan_sha256=sha256(d / 'candidate_plan.json'),
                   steps=plan['training']['steps'], schedule_steps=plan['training']['schedule_steps'],
                   EMA=plan['training']['EMA'], windows=plan['sampling']['windows'],
                   preflight={k: pf[k] for k in ('status', 'train_cases', 'windows', 'predicted_batch_digest', 'architecture')})
        if self.full_release():
            hc = self.report['historical_comparison']
            hc['s6_steps'] = compare(plan['training']['steps'], HISTORICAL['s6_steps'])
            hc['s6_predicted_batch_digest'] = compare(pf['predicted_batch_digest'], HISTORICAL['s6_batch_digest'])
        train = self.rebind(T6, d)
        if self.smoke:
            self.gpu_step('s6_c758h', train)
            self.stage('s6_c758h', status='PLAN_GENERATED_PREFLIGHT_OK__TRAIN_DRY_RUN')
            return d
        if not trained:
            self.run('s6_c758h_train', train)
        report = read_json(t / 'report.json')
        steps = plan['training']['steps']
        if report['status'] != 'COMPLETE' or report['steps_actual'] != steps:
            raise StageError('S6 training report not COMPLETE')
        ema = read_json(t / 'ema' / 'report.json')
        self.stage('s6_c758h', status='COMPLETE', batch_digest=ema['batch_digest'],
                   batch_digest_equals_preflight=ema['batch_digest'] == pf['predicted_batch_digest'],
                   ema_endpoints={k: {'sha256': v['sha256'], 'state_hash': v['state_hash']} for k, v in ema['endpoints'].items()},
                   resume=rel(t / f'resume_{steps}.pt'), resume_sha256=sha256(t / f'resume_{steps}.pt'),
                   training_loop_seconds=report.get('training_loop_seconds'))
        if self.full_release():
            hc = self.report['historical_comparison']
            hc['s6_batch_digest'] = compare(ema['batch_digest'], HISTORICAL['s6_batch_digest'])
            for key, name in (('s6_ema_point', 'model.pt'), ('s6_ema_center', 'center_head.pt'),
                              ('s6_ema_radius', 'radius_head.pt'), ('s6_ema_hidden', 'hidden_head.pt'),
                              ('s6_ema_observed', 'observed_center_head.pt')):
                hc[key] = compare(ema['endpoints'][name]['state_hash'], HISTORICAL[key])
        self.seal('s6_c758h', [d / 'candidate_plan.json', t / 'report.json', t / 'job.json', t / f'resume_{steps}.pt',
                               t / 'ema' / 'model.pt'])
        return d

    # ---------------------------------------------------------------- S7
    def s7_plan(self, d6, resume):
        p6 = read_json(d6 / 'candidate_plan.json')
        steps0, schedule = p6['training']['steps'], p6['training']['schedule_steps']
        end = S7_ENDPOINT_FRACTION[0] * schedule // S7_ENDPOINT_FRACTION[1]         # 13848 for the release
        training = dict(p6['training'])
        training.update(steps=end, phase_energy_normalized=True, resume_step=steps0,
                        endpoint='Selected common horizon endpoint; original EMA unchanged',
                        tau=f'Original cosine/tau horizon{schedule} continued; no reset')
        plan = dict(p6)
        pending = 'PENDING_S6_OUTPUT'
        plan.update(candidate='C787H', training=training, resume_step=steps0, middle_step=end,
                    resume_checkpoint=rel(resume) if resume else pending,
                    resume_checkpoint_sha256=sha256(resume) if resume else pending,
                    baseline_training=rel(resume.parent) if resume else pending,
                    source_pins={**p6['source_pins'], T7: sha256(ROOT / T7)},
                    policy='C784R endpoint rule inlined: end=3*schedule_steps//4; S5 runtime; resume from S6 checkpoint')
        return plan, steps0, end

    def stage7(self, rt5, d6):
        d = self.out / 's7_c787h'
        t = d / 'training_candidate'
        if self.sealed('s7_c787h'):
            return d / 'runtime'
        steps0 = read_json(d6 / 'candidate_plan.json')['training']['steps']
        resume, standin = d6 / 'training_candidate' / f'resume_{steps0}.pt', None
        if self.smoke and not self.history:
            plan, steps0, end = self.s7_plan(d6, None)
            d.mkdir()
            write_json(d / 'candidate_plan.dryrun.json', plan)
            check = self.run('s7_rebind_import', self.rebind(T7, d, 'import'), cpu=True)
            self.gpu_step('s7_c787h', self.rebind(T7, d))
            self.stage('s7_c787h', status='PLAN_GENERATOR_DRY_RUN (resume fields need S6 outputs; no historical S6 stand-in)',
                       resume_step=steps0, end_step=end, rebind=self.last_json(check))
            return None
        if self.smoke:
            standin = resume = ROOT / HIST_C758_TRAINING / f'resume_{steps0}.pt'   # read-only historical S6 stand-in
        plan, steps0, end = self.s7_plan(d6, resume)
        if d.exists():
            if not (d / 'candidate_plan.json').exists() or read_json(d / 'candidate_plan.json') != plan:
                raise StageError(f'{rel(d)} holds a different plan; inspect and move it aside')
            if (t.exists() and not (t / 'report.json').exists()) or (d / 'runtime').exists():
                raise StageError(f'{rel(d)} holds a partial run; move the partial part aside, then --resume')
        else:
            d.mkdir()
            write_json(d / 'candidate_plan.json', plan)
        trained = (t / 'report.json').exists()
        if not trained:
            check = self.run('s7_rebind_check', self.rebind(T7, d, 'plan'), cpu=True)
            self.run('s7_preflight', [self.py, DRIVER / 'preflight.py', 's7', '--stage-dir', d], cpu=True)
            self.stage('s7_c787h', rebind=self.last_json(check))
        pf = read_json(d / 'preflight.json')
        self.stage('s7_c787h', plan=rel(d / 'candidate_plan.json'), plan_sha256=sha256(d / 'candidate_plan.json'),
                   resume_step=steps0, end_step=end, s6_stand_in=rel(standin) if standin else None,
                   preflight={k: pf[k] for k in ('status', 'new_updates', 'predicted_suffix_batch_digest',
                                                 'baseline_job_checks', 'resume_plan_sha256_matches_s6_plan')})
        if self.full_release():
            self.report['historical_comparison']['s7_steps'] = compare(end, HISTORICAL['s7_steps'])
        train = self.rebind(T7, d)
        if self.smoke:
            self.gpu_step('s7_c787h', train)
            self.stage('s7_c787h', status='PLAN_GENERATED_PREFLIGHT_OK__TRAIN_DRY_RUN (S6 = historical stand-in)')
            return None
        if not trained:
            self.run('s7_c787h_train', train)
        report = read_json(t / 'report.json')
        if report['status'] != 'COMPLETE' or report['steps_actual'] != end:
            raise StageError('S7 training report not COMPLETE')
        self.run('s7_c787h_package', [self.py, DRIVER / 'c787h_package.py', '--runtime', rt5, '--stage-dir', d], cpu=True)
        audit = read_json(d / 'package_audit.json')
        self.stage('s7_c787h', status='COMPLETE', runtime=audit['runtime'], runtime_members=audit['runtime_members'],
                   deployed_states=audit['deployed_states'], archive=audit['archive'], archive_sha256=audit['archive_sha256'],
                   suffix_batch_digest=audit['suffix_batch_digest'],
                   suffix_digest_equals_preflight=audit['suffix_batch_digest'] == pf['predicted_suffix_batch_digest'],
                   training_loop_seconds=report.get('training_loop_seconds'))
        if self.full_release():
            hc = self.report['historical_comparison']
            for key, name in (('s7_center', 'center_head.pt'), ('s7_radius', 'radius_head.pt'),
                              ('s7_hidden', 'hidden_head.pt'), ('s7_observed', 'observed_center_head.pt')):
                hc[key] = compare(audit['deployed_states'][name], HISTORICAL[key])
            hc['s7_point_is_s6_ema_point'] = compare(audit['deployed_states']['model.pt'], HISTORICAL['s6_ema_point'])
        runtime = d / 'runtime'
        self.seal('s7_c787h', [d / 'candidate_plan.json', d / 'package_audit.json', d / 'deployed_checkpoint.pt',
                               *[p for p in runtime.rglob('*') if p.is_file()]])
        return runtime

    # ---------------------------------------------------------------- smoke-only parity checks
    def smoke_parity_s5(self):
        """c728p_convert on the historical C628 parent (read-only) vs the historical C728P outputs."""
        hist = ROOT / HIST_C628_PARENT
        rt = self.stage5({n: hist / n for n in COMPONENT_FILES}, name='s5_parity', cases_file=self.out / 'cases81.json')
        init = read_json(rt.parent / 'initialization_report.json')
        old = read_json(ROOT / HIST_C728_DIR / 'initialization_report.json')
        old_rt = ROOT / HIST_C728_DIR / 'common_runtime'
        result = {'initial_states': init['initial_states'] == old['initial_states'],
                  'train_windows': init['train_windows'] == old['train_windows'],
                  'identity_checks': init['CPU_identity_checks'] == old['CPU_identity_checks'],
                  'members_except_model_pt_bytes': all(v == sha256(old_rt / k) for k, v in init['runtime_members'].items()
                                                       if k != 'model.pt'),
                  'model_pt_bytes (expected False: new C728 record)': init['runtime_members']['model.pt'] == sha256(old_rt / 'model.pt')}
        self.report['historical_comparison']['s5_parity'] = result
        self.stage('s5_parity', comparison=result)
        return rt

    def smoke_parity_package(self, rt5):
        """c787h_package on the historical C787R training outputs (read-only) vs the C787H ZIP."""
        import hashlib
        import zipfile
        out = self.out / 's7_parity_package'
        self.run('s7_parity_package', [self.py, DRIVER / 'c787h_package.py', '--runtime', rt5, '--stage-dir',
                                       ROOT / HIST_C787_DIR, '--out', out], cpu=True)
        audit = read_json(out / 'package_audit.json')
        with zipfile.ZipFile(ROOT / HIST_C787H_ZIP) as z:
            historical = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
        old_audit = read_json(ROOT / HIST_C787_DIR / 'package_audit.json')['archives']['candidate']
        result = {'all_15_runtime_members_byte_identical_to_C787H_zip': audit['runtime_members'] == historical,
                  'mismatched_members': sorted(k for k in historical if audit['runtime_members'].get(k) != historical[k]),
                  'deployed_checkpoint_byte_identical': audit['deployed_checkpoint_sha256'] == old_audit['checkpoint_sha256'],
                  'head_states_equal_spec': all(audit['deployed_states'][f] == HISTORICAL[k] for k, f in
                                                (('s7_center', 'center_head.pt'), ('s7_radius', 'radius_head.pt'),
                                                 ('s7_hidden', 'hidden_head.pt'), ('s7_observed', 'observed_center_head.pt'))),
                  'point_state_equals_s6_ema': audit['deployed_states']['model.pt'] == HISTORICAL['s6_ema_point']}
        self.report['historical_comparison']['s7_package_parity'] = result
        self.stage('s7_parity_package', status='RAN_ON_HISTORICAL_S7_OUTPUTS (CPU)', comparison=result)

    # ---------------------------------------------------------------- main flow
    def execute(self):
        self.prologue()
        self.resolve_holdout()
        components = self.stage1_4()
        if self.smoke:
            s1 = self.smoke_stage1_cpu()
            heads = self.out / 'smoke_standin_heads'
            self.run('smoke_standin_heads', [self.py, DRIVER / 'preflight.py', 'standins', '--joint-head',
                                             s1.parent / 'final_head.pt', '--out', heads], cpu=True)
            standins = {'model.pt': s1, **{h: heads / h for h in HEAD_FILES}}
            self.stage('smoke_stand_ins', s5_parent={k: rel(v) for k, v in standins.items()},
                       note='S1 = fresh 1-step CPU point; S2-S4 = UNTRAINED heads in the exact payload formats '
                            '(S2-S4 training hard-codes CUDA). No historical weight is used by this chain.')
            rt5 = self.stage5(standins)
            self.history = self.full_release() and all((ROOT / p).exists() for p in HIST_ARTIFACTS)
            if self.history:
                rt5 = self.smoke_parity_s5()      # S6/S7 smoke continue from the historical-equivalent runtime
            else:
                self.stage('smoke_parity', status='SKIPPED (fold, or historical artifacts absent)')
        else:
            rt5 = self.stage5(components)
        d6 = self.stage6(rt5)
        runtime = self.stage7(rt5, d6)
        if self.smoke and self.history:
            self.smoke_parity_package(rt5)
        return runtime

    def finish(self, status, runtime=None, error=None):
        self.report['status'] = status
        self.report['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
        if self.holdout is not None:
            self.report['holdout'] = {k: self.holdout[k] for k in ('mode', 'requested', 'counts')}
        if runtime is not None:
            self.report['latcadf_runtime'] = rel(runtime)
        if error is not None:
            self.report['error'] = error
        mism = [k for k, v in self.report['historical_comparison'].items() if isinstance(v, dict) and v.get('match') is False]
        self.report['historical_mismatches_recorded'] = mism
        if self.owned:
            write_json(self.out / 'rebuild_report.json', self.report)
        self.log(f'{status} elapsed={self.report["elapsed_seconds"]} s; recorded historical mismatches: {mism or "none"}')


def summarize_smoke(driver):
    r = driver.report
    driver.log('SMOKE SUMMARY (CPU only; nothing ran on the GPU)')
    for name, values in r['stages'].items():
        driver.log(f'  stage {name}: {values.get("status", "")}')
    for key, value in r['historical_comparison'].items():
        driver.log(f'  parity {key}: {json.dumps(value)[:400]}')
    for step in r['gpu_steps']:
        cmd = step['command']
        driver.log(f'  GPU needed: {step["stage"]} est {step["estimate_seconds"][0]}-{step["estimate_seconds"][1]} s: '
                   + ' '.join(Path(c).name if '/' in c else c for c in cmd[1:4]) + ' ...')


def cleanup_smoke(driver):
    for path in sorted(driver.out.iterdir()):
        if path.name != 'driver.log':
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='new directory under runs/FINAL_REBUILD/')
    parser.add_argument('--holdout-re', help="comma list of Reynolds numbers and/or one '>=X' term")
    parser.add_argument('--holdout-aoa', help='comma list of angles of attack')
    parser.add_argument('--smoke', action='store_true', help='CPU-only plumbing check; GPU stages are dry-run')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted run in --out (sealed stages reused)')
    parser.add_argument('--keep-smoke-outputs', action='store_true', help='keep smoke outputs (default: keep driver.log only)')
    parser.add_argument('--python', help='interpreter for all stage processes (default: this interpreter)')
    parser.add_argument('--c675-max-wall-seconds', type=float, default=C675_MAX_WALL)
    parser.add_argument('--c675-max-output-gib', type=float, default=C675_MAX_GIB)
    args = parser.parse_args(argv)
    driver = Driver(args)
    status, runtime, error = 'FAILED', None, None
    try:
        runtime = driver.execute()
        status = 'SMOKE_COMPLETE' if driver.smoke else 'COMPLETE'
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
                if not args.keep_smoke_outputs:
                    cleanup_smoke(driver)
    return 0 if status != 'FAILED' else 1


if __name__ == '__main__':
    sys.exit(main())
