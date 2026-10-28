"""CPU-only preflight of a generated stage-6/7 plan (no GPU, no training, no data arrays read).

Re-executes the checks the historical train.py makes before/around its GPU work so that a
plan error fails in seconds instead of after GPU time:
  train.py:25 source pins; :27-30 runtime import and the two hidden.detach() calls;
  :31-34 model/head loading and initial states; :46-47 case/window inventory (metadata only);
  :57,61-62 the seeded sampler (replayed on CPU -> batch digest, coverage of :91-92);
  C787R :59-64 resume checkpoint, state-key compatibility, optimizer step, baseline job.json.
Also: `hashes` records sha256 + state hash of checkpoint files (records only), and `standins`
(smoke only) writes untrained heads in the exact S2-S4 payload formats.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXCLUDED_REAL, HEAD_KEYS, ROOT, read_json, sd_hash, sha256, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / 'src'))


def _runtime(plan):
    import torch
    from realpde import identity_interval_hardening as M
    from realpde.hidden_sps_interval import load_hidden_residual
    from realpde.sps_radius_capacity import load_expanded_head
    rt = ROOT / plan['runtime']
    spec = importlib.util.spec_from_file_location('c716g_point_preflight', rt / 'factorized_model.py')
    F = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = F
    spec.loader.exec_module(F)
    detaches = textwrap.dedent(inspect.getsource(F.FactorizedFlowOperator.forward)).count('hidden.detach()')
    assert detaches == 2, 'train.py:29 expects exactly two hidden.detach() calls'
    cp = torch.load(rt / 'model.pt', map_location='cpu', weights_only=False)
    point = F.FactorizedFlowOperator(F.ModelConfig.from_dict(cp['model_config']))
    point.load_state_dict(cp['state_dict'])
    heads = torch.nn.ModuleDict({k: M.load_head(rt / f, 'cpu') if k == 'center' else load_expanded_head(rt / f, 'cpu')
                                 if k == 'radius' else load_hidden_residual(rt / f, 'cpu') for k, f in HEAD_KEYS.items()})
    initial = {'point': M.state_hash(point), **{k: M.state_hash(v) for k, v in heads.items()}}
    dropouts = [float(m.p) for m in point.modules() if isinstance(m, torch.nn.Dropout)]
    return point, heads, initial, {'hidden_detach_calls': detaches, 'dropout_modules': len(dropouts),
                                   'dropout_p': sorted(set(dropouts))}


def _windows(plan):
    from realpde.data import discover_cases
    train, evaluation = plan['train_cases'], plan['eval_cases']
    assert EXCLUDED_REAL not in train and not set(train) & set(evaluation)
    cases = [c for c in discover_cases(str(ROOT / 'data/extracted/train_real'), exclude_files=[EXCLUDED_REAL])
             if c.path.name in train]
    assert [c.path.name for c in cases] == train, 'discover order / membership differs from plan train_cases'
    s = plan['sampling']
    windows = [(i, st) for i, c in enumerate(cases) for st in range(0, c.frames - s['support_frames'] + 1, s['stride'])]
    assert len(windows) == s['windows'], f"window count {len(windows)} != plan {s['windows']}"
    return cases, windows


def _replay(n, rng, order, cursor, first, last):
    import torch
    digest, seen = hashlib.sha256(), set()
    for _ in range(first, last + 1):
        if cursor >= n:
            order = torch.randperm(n, generator=rng)
            cursor = 0
        ix = order[cursor:cursor + 8]
        cursor += len(ix)
        seen.update(ix.tolist())
        digest.update(ix.numpy().tobytes())
    return digest.hexdigest(), seen, order, cursor


def s6(stage_dir):
    import torch
    plan = read_json(Path(stage_dir) / 'candidate_plan.json')
    bad = [p for p, h in plan['source_pins'].items() if sha256(ROOT / p) != h]
    assert not bad, 'source pins changed: ' + ', '.join(bad[:5])
    tr = plan['training']
    steps, sched = tr['steps'], tr['schedule_steps']
    assert sched == 2 * steps and tr['EMA'] == .999 ** (6000 / steps)
    point, heads, initial, arch = _runtime(plan)
    assert initial == plan['initial_states'], 'runtime states differ from the S5 initialization record'
    cases, windows = _windows(plan)
    rng = torch.Generator().manual_seed(tr['seed'] + 1)                       # train.py:57
    order = torch.randperm(len(windows), generator=rng)
    digest, seen, _, _ = _replay(len(windows), rng, order, 0, 1, steps)
    assert {cases[windows[i][0]].path.name for i in seen} == set(plan['train_cases'])   # train.py:91
    assert len(seen) == min(len(windows), steps * 8)                                     # train.py:92
    result = {'status': 'PREFLIGHT_OK', 'stage': 's6', 'train_cases': len(cases), 'windows': len(windows),
              'steps': steps, 'predicted_batch_digest': digest, 'unique_windows_after_steps': len(seen),
              'initial_states': initial, 'architecture': arch, 'gpu_required': True}
    write_json(Path(stage_dir) / 'preflight.json', result)
    return result


def s7(stage_dir):
    import torch
    plan = read_json(Path(stage_dir) / 'candidate_plan.json')
    bad = [p for p, h in plan['source_pins'].items() if sha256(ROOT / p) != h]
    assert not bad, 'source pins changed: ' + ', '.join(bad[:5])
    tr = plan['training']
    end, resume_step, sched = tr['steps'], plan['resume_step'], tr['schedule_steps']
    assert tr['phase_energy_normalized'] is True and plan['middle_step'] == end and resume_step < end <= sched
    point, heads, initial, arch = _runtime(plan)
    cases, windows = _windows(plan)
    resume = ROOT / plan['resume_checkpoint']
    assert sha256(resume) == plan['resume_checkpoint_sha256']                        # C787R train.py:59
    saved = torch.load(resume, map_location='cpu', weights_only=False)
    assert saved['step'] == resume_step and saved['scheduler']['last_epoch'] == resume_step
    assert saved['unique_sampled_windows'] == len(windows)                              # :60
    point.load_state_dict(saved['point'])                                              # :60 key compatibility
    heads.load_state_dict(saved['heads'])
    point.load_state_dict(saved['ema_point'])
    heads.load_state_dict(saved['ema_heads'])
    assert {int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v} == {resume_step}   # :62
    oldjob = read_json(ROOT / plan['baseline_training'] / 'job.json')                  # :63-64 (CPU part)
    sample_ids = [[cases[i].path.name, s] for i, s in windows]
    job_checks = {'sample_ids': oldjob['sample_ids'] == sample_ids,
                  'initial_states': oldjob['initial_states'] == initial,
                  'train_h5_sha256': 'checked by train.py on the GPU run'}
    assert job_checks['sample_ids'] and job_checks['initial_states'], job_checks
    rng = torch.Generator()
    rng.set_state(saved['rng'])
    digest, _, _, _ = _replay(len(windows), rng, saved['sampler_order'], saved['sampler_cursor'], resume_step + 1, end)
    s6_plan = ROOT / plan['baseline_training'] / '..' / 'candidate_plan.json'
    result = {'status': 'PREFLIGHT_OK', 'stage': 's7', 'train_cases': len(cases), 'windows': len(windows),
              'resume_step': resume_step, 'end_step': end, 'new_updates': end - resume_step,
              'predicted_suffix_batch_digest': digest, 'resume_prior_batch_digest': saved['batch_digest'],
              'resume_plan_sha256_matches_s6_plan': saved['plan_sha256'] == sha256(s6_plan) if s6_plan.exists() else None,
              'baseline_job_checks': job_checks, 'architecture': arch, 'gpu_required': True}
    write_json(Path(stage_dir) / 'preflight.json', result)
    return result


def standins(joint_head, out):
    """Smoke only: UNTRAINED heads in the exact S2-S4 payload formats (S2-S4 training needs CUDA).

    center = the S1 joint head; radius = ExpandedFastIntervalHead.expand(joint head, seed 20260907579)
    as fixed_sps_capacity_training does; hidden/observed = fresh HiddenIntervalResidual(128/168)
    with the zero last layer of their constructors."""
    import torch
    from realpde.hidden_sps_interval import HiddenIntervalResidual
    from realpde.identity_interval_hardening import load_head
    from realpde.sps_radius_capacity import ExpandedFastIntervalHead
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = {'smoke_stand_in': True, 'trained': False, 'source': str(joint_head)}
    head = load_head(joint_head, 'cpu')
    torch.save({'model_config': head.config.to_dict(), 'state_dict': head.state_dict(), 'metadata': meta},
               out / 'center_head.pt')
    radius = ExpandedFastIntervalHead.expand(head, seed=20260907579)
    torch.save({'architecture': radius.architecture, 'model_config': radius.config.to_dict(),
                'state_dict': radius.state_dict(), 'metadata': meta}, out / 'radius_head.pt')
    for name, channels in (('hidden_head.pt', 128), ('observed_center_head.pt', 168)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(channels)
            model = HiddenIntervalResidual(channels, 20)
        torch.save({'architecture': model.architecture, 'model_config': {'channels': channels, 'out_steps': 20},
                    'state_dict': model.state_dict(), 'metadata': meta}, out / name)
    return {name: str(out / name) for name in ('center_head.pt', 'radius_head.pt', 'hidden_head.pt', 'observed_center_head.pt')}


def hashes(paths):
    import torch
    out = {}
    for p in paths:
        payload = torch.load(ROOT / p, map_location='cpu', weights_only=False)
        out[p] = {'sha256': sha256(ROOT / p), 'state_hash': sd_hash(payload['state_dict'])}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('s6', 's7'):
        sub.add_parser(name).add_argument('--stage-dir', type=Path, required=True)
    h = sub.add_parser('hashes')
    h.add_argument('--paths-json', type=Path, required=True)
    h.add_argument('--out', type=Path, required=True)
    st = sub.add_parser('standins')
    st.add_argument('--joint-head', type=Path, required=True)
    st.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if args.command == 'standins':
        print(json.dumps({'status': 'STANDINS_WRITTEN', 'files': sorted(standins(args.joint_head, args.out))}))
        return
    if args.command == 'hashes':
        write_json(args.out, hashes(read_json(args.paths_json)))
        print(json.dumps({'status': 'HASHED', 'files': len(read_json(args.paths_json))}))
        return
    result = (s6 if args.command == 's6' else s7)(args.stage_dir)
    print(json.dumps({k: result[k] for k in ('status', 'stage', 'train_cases', 'windows')}))


if __name__ == '__main__':
    main()
