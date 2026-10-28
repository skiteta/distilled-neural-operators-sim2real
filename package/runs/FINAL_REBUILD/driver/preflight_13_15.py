"""CPU-only tools for the stage 13-15 driver (no GPU, no training, no optimizer update).

  replay   replays every seeded stream of a generated stage 13-15 plan exactly as the historical trainers draw them:
           train_simat.py / train_simati_focus*.py:176-222 (per phase: source windows seed'+1/+31, reflection +2/+32,
           lattice phase +3/+33, cadence +4/+34 with seed' = seed + stream_seed_offset; distillation phase first when
           warmup_steps > 0; checkpoint digests at the distillation end and at fine-tune steps fine//2 and fine) and the
           simulation stream of the plan's simulation module, one call per fine-tune step: sim_replay_all.py:33-52
           (seed+19; preflight_8_12.replay_sim), sim_replay_focus.py:36-67 (focus pool seed+29, other pool seed+19,
           round(8f) focus windows per batch) or sim_replay_focus2.py (per-window emphasis ratio, accumulator).
  states   the trainers' model set-up on CPU (train_simati_focus.py:52-79 == train_simat.py:48-74): teacher from the plan
           runtime, the two hidden.detach() removed, student depth/block surgery, the four heads; returns the state hashes
           the training writes to job.json (teacher_point_hash, initial_states) plus architecture facts.
  runtime-input  a few real training windows (released data, first training case) as an [N,20,32,64,3] float32 array
           for the CPU runtime smoke of a packaged runtime (runtime_smoke_13_15.py).
Other tools (inventory, extract, ZIP compare) are the stage 8-12 ones (preflight_8_12.py), called unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_json, write_json  # noqa: E402
from common_13_15 import SIM_ALL, SIM_FOCUS, SIM_FOCUS2, SIM_KIND, focus_per_batch  # noqa: E402
import preflight_8_12  # noqa: E402  (replay_sim, _windows: the verified stage 8-13 replay code)

sys.path.insert(0, str(ROOT / 'src'))


def replay_phases(frames, training):
    import torch
    windows, lengths = preflight_8_12._windows(frames), [f for _, f in frames]
    n, p = len(windows), float(training['cadence2_probability'])
    lp = list(training.get('lattice_phases', [0, 1, 2, 3]))
    assert lp and all(q in (0, 1, 2, 3) for q in lp)
    seed = training['seed'] + int(training.get('stream_seed_offset', 0))          # train_simati_focus.py:177
    warm, fine = int(training['warmup_steps']), int(training['finetune_steps'])
    assert warm + fine == int(training['steps'])
    phases, checkpoints, global_step = [], {}, 0
    for phase, steps in [(ph, st) for ph, st in [('distillation', warm), ('finetune', fine)] if st > 0]:
        dist = phase == 'distillation'
        rng = torch.Generator().manual_seed(seed + (31 if dist else 1))
        ag = torch.Generator().manual_seed(seed + (32 if dist else 2))
        prng = torch.Generator().manual_seed(seed + (33 if dist else 3))
        crng = torch.Generator().manual_seed(seed + (34 if dist else 4))
        order, cursor, seen = torch.randperm(n, generator=rng), 0, set()
        digest, adigest = hashlib.sha256(), hashlib.sha256()
        samples = reflected = fallbacks = 0
        lat, cadc = [0] * 4, [0, 0]
        for step in range(1, steps + 1):
            if cursor >= n:
                order, cursor = torch.randperm(n, generator=rng), 0
            ix = order[cursor:cursor + 8]
            cursor += len(ix)
            seen.update(ix.tolist())
            digest.update(ix.numpy().tobytes())
            phs = [lp[q % len(lp)] for q in torch.randint(4, (len(ix),), generator=prng).tolist()]
            req = (torch.rand(len(ix), generator=crng) < p).long().tolist()
            cads = [c if (c == 0 or windows[i][1] + 79 <= lengths[windows[i][0]]) else 0 for i, c in zip(ix.tolist(), req)]
            fallbacks += sum(1 for r, a in zip(req, cads) if r == 1 and a == 0)
            for q in phs:
                lat[q] += 1
            for c in cads:
                cadc[c] += 1
            flip = torch.rand(len(ix), generator=ag) < .5
            adigest.update(flip.numpy().tobytes())
            samples += len(ix)
            reflected += int(flip.sum())
            global_step += 1
            if (dist and step == steps) or (not dist and step in {fine // 2, fine}):        # :216 resume checkpoints
                checkpoints[str(global_step)] = {'phase': phase, 'phase_step': step, 'batch_digest': digest.hexdigest(),
                                                 'augmentation_digest': adigest.hexdigest(), 'unique_sampled_windows': len(seen),
                                                 'source_samples': samples, 'reflected_samples': reflected}
        if not dist:
            assert len(seen) == n                                                          # :223
        phases.append({'phase': phase, 'steps': steps, 'unique_sampled_windows': len(seen), 'batch_digest': digest.hexdigest(),
                       'augmentation_digest': adigest.hexdigest(), 'source_samples': samples, 'lattice_phase_counts': lat,
                       'cadence_counts': cadc, 'cadence_trailing_fallbacks': fallbacks, 'cadence2_probability': p,
                       'reflected_samples': reflected})
    return {'stream_seed': seed, 'windows': n, 'steps': global_step, 'phases': phases, 'checkpoints': checkpoints}


def replay_sim_focus(sim_frames, training, calls, train_cases, mode):
    """sim_replay_focus.py:36-67 ('focus', fraction) / sim_replay_focus2.py (same pools, accumulator k per batch)."""
    import torch
    windows = preflight_8_12._windows(sim_frames)
    names = [name for name, _ in sim_frames]
    focus_names = sorted(set(names) - set(train_cases))                  # :38 sims whose real counterpart is not trained on
    fset = {i for i, name in enumerate(names) if name in set(focus_names)}
    focus_pool = torch.tensor([p for p, (i, s) in enumerate(windows) if i in fset], dtype=torch.long)
    other_pool = torch.tensor([p for p, (i, s) in enumerate(windows) if i not in fset], dtype=torch.long)
    assert len(focus_pool) > 0 and len(other_pool) > 0
    seed = training['seed']                                              # plan seed, no stream offset (:16)
    rng, frng = torch.Generator().manual_seed(seed + 19), torch.Generator().manual_seed(seed + 29)
    if mode == 'focus':
        fraction = float(training.get('sim_focus_fraction', 0.0))
        assert 0.0 < fraction < 1.0
        k_fixed, acc = focus_per_batch(fraction), None
    else:
        ratio = float(training['sim_focus_ratio'])
        assert ratio >= 1.0
        fraction = ratio * len(focus_pool) / (ratio * len(focus_pool) + len(other_pool))
        k_fixed, acc = None, 0.0
    forder, fcursor = focus_pool[torch.randperm(len(focus_pool), generator=frng)], 0
    order, cursor = other_pool[torch.randperm(len(other_pool), generator=rng)], 0
    digest, seen, samples, focus_samples, first = hashlib.sha256(), set(), 0, 0, None
    for call in range(calls):
        if acc is None:
            k = k_fixed
        else:
            prev = acc
            acc += 8 * fraction
            k = max(0, min(8, int(acc) - int(prev)))
        if cursor + (8 - k) > len(order):
            order, cursor = other_pool[torch.randperm(len(other_pool), generator=rng)], 0
        if fcursor + k > len(forder):
            forder, fcursor = focus_pool[torch.randperm(len(focus_pool), generator=frng)], 0
        indices = torch.cat((forder[fcursor:fcursor + k], order[cursor:cursor + 8 - k]))
        fcursor += k
        cursor += 8 - k
        seen.update(indices.tolist())
        digest.update(indices.numpy().tobytes())
        samples += len(indices)
        focus_samples += k
        if call == 0:
            first = [[names[windows[i][0]], windows[i][1]] for i in indices.tolist()]
    return {'policy_cases': len(sim_frames), 'windows': len(windows), 'calls': calls, 'samples': samples,
            'unique_windows': len(seen), 'digest': digest.hexdigest(), 'first_sample_ids': first, 'mode': mode,
            'focus': {'fraction': fraction, 'per_batch': k_fixed if acc is None else 'accumulator', 'cases': focus_names,
                      'focus_windows': len(focus_pool), 'other_windows': len(other_pool), 'focus_samples': focus_samples}}


def replay(plan, inv, sim_module):
    tr = plan['training']
    out = replay_phases(inv['real'], tr)
    assert out['windows'] == plan['sampling']['windows'], 'replayed window count differs from the plan'
    if tr.get('sim_replay'):
        mode = SIM_KIND[sim_module]
        calls = int(tr['finetune_steps'])                                  # SIMR.loss only in the fine-tune phase (:131)
        if mode == 'all':
            sim = preflight_8_12.replay_sim(inv['sim'], tr, calls)
            sim['mode'] = 'all'
        else:
            sim = replay_sim_focus(inv['sim'], tr, calls, plan['train_cases'], mode)
        out['simulation_replay'] = sim
    return out


def initial_states(plan):
    """train_simati_focus.py:47-79 (== train_simat.py:43-74) on CPU; values as written to job.json/report.json."""
    import copy
    import importlib.util
    import inspect
    import textwrap
    import torch
    from realpde import identity_interval_hardening as M
    from realpde.hidden_sps_interval import load_hidden_residual
    from realpde.sps_radius_capacity import load_expanded_head
    tr = plan['training']
    for key in ('physics_features', 'student_dropout', 'head_refit', 'no_multiscale'):
        assert not tr.get(key), f'{key} is not part of the stage 13-15 recipe'
    torch.manual_seed(tr['seed'])
    device = torch.device('cpu')
    rt = ROOT / plan['runtime']
    spec = importlib.util.spec_from_file_location('depth_teacher_factorized', rt / 'factorized_model.py')
    F = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = F
    spec.loader.exec_module(F)
    source = textwrap.dedent(inspect.getsource(F.FactorizedFlowOperator.forward))
    assert source.count('hidden.detach()') == 2
    ns = dict(F.__dict__)
    exec(source.replace('hidden.detach()', 'hidden'), ns)
    F.FactorizedFlowOperator.forward = ns['forward']
    cp = torch.load(rt / 'model.pt', map_location='cpu', weights_only=False)
    teacher = F.FactorizedFlowOperator(F.ModelConfig.from_dict(cp['model_config'])).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(cp['state_dict'])
    teacher_hash = M.state_hash(teacher)
    config = dict(cp['model_config'])
    assert config['depth'] == 10
    depth = int(tr.get('student_depth', 4))
    config['depth'] = depth
    point = F.FactorizedFlowOperator(F.ModelConfig.from_dict(config)).to(device).train().requires_grad_(True)
    kept = tr['retained_teacher_blocks']
    assert len(kept) == depth and kept == sorted(set(kept)) and max(kept) < 10
    initial = {}
    for name in point.state_dict():
        old = name
        if name.startswith('blocks.'):
            items = name.split('.')
            items[1] = str(kept[int(items[1])])
            old = '.'.join(items)
        initial[name] = cp['state_dict'][old]
    point.load_state_dict(initial)
    files = {'center': 'center_head.pt', 'radius': 'radius_head.pt', 'hidden': 'hidden_head.pt', 'observed': 'observed_center_head.pt'}
    heads = torch.nn.ModuleDict({k: M.load_head(rt / f, device) if k == 'center' else load_expanded_head(rt / f, device)
                                 if k == 'radius' else load_hidden_residual(rt / f, device) for k, f in files.items()}).eval().requires_grad_(True)
    teacher_heads = copy.deepcopy(heads)
    dropout = [v for v in point.modules() if isinstance(v, torch.nn.Dropout)]
    assert len(dropout) == depth + 2 and all(abs(v.p - config['dropout']) < 1e-12 and v.training for v in dropout)
    return {'status': 'STATES_OK', 'teacher_point_hash': teacher_hash,
            'initial_states': {'point': M.state_hash(point), 'heads': M.state_hash(heads)},
            'teacher_heads_equal_initial_heads': M.state_hash(teacher_heads) == M.state_hash(heads),
            'teacher_model_config': cp['model_config'], 'student_model_config': config, 'kept_blocks': kept,
            'student_parameters': sum(p.numel() for p in point.parameters()),
            'teacher_parameters': sum(p.numel() for p in teacher.parameters()),
            'dropout_modules': len(dropout), 'dropout_p': sorted({v.p for v in dropout})}


def runtime_input(train_cases, count):
    import numpy as np
    from realpde.data import FlowWindowDataset, discover_cases
    cases = [c for c in discover_cases(str(ROOT / 'data/extracted/train_real'), exclude_files=['7575_0.h5'])
             if c.path.name in set(train_cases)]
    case = cases[0]
    data = FlowWindowDataset([case], stride=1)
    starts = [round(i * (len(data) - 1) / max(1, count - 1)) for i in range(count)]
    x = np.stack([data[s][0].numpy() for s in starts]).astype(np.float32)
    assert x.shape == (count, 20, 32, 64, 3), x.shape
    return x, {'case': case.path.name, 'starts': starts, 'shape': list(x.shape),
               'sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('replay')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--sim-module', choices=[SIM_ALL, SIM_FOCUS, SIM_FOCUS2], required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('states')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('runtime-input')
    p.add_argument('--cases-json', type=Path, required=True)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--out', type=Path, required=True, help='.npy file; a .json record is written next to it')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if args.command == 'replay':
        result = replay(read_json(args.plan), read_json(args.inventory), args.sim_module)
    elif args.command == 'states':
        result = initial_states(read_json(args.plan))
    else:
        import numpy as np
        x, result = runtime_input(read_json(args.cases_json), args.count)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, x)
        args.out = args.out.with_suffix('.json')
    write_json(args.out, result)
    print(json.dumps({'status': 'OK', 'command': args.command, 'out': str(args.out)}))


if __name__ == '__main__':
    main()
