"""CPU-only tools for the stage 8-13 driver (no GPU, no training, no data arrays read except by `runtime`).

  inventory  frame counts of the plan's real training cases and of the simulation list (h5 metadata only),
             windows, and the S6 exposure base ceil(6000*windows/42847).
  replay     replays a generated plan's seeded training streams exactly as the historical scripts draw them:
             LATCADF train.py:74-89 (source windows seed+1, reflection seed+2, lattice phase seed+3, cadence seed+4,
             trailing-start cadence fallback from frame counts) or train_sd10.py/train_cycle4.py/train_sim.py:137-151
             (the same four generators at seed+stream_seed_offset) and sim_replay_all.py:33-52 (simulation window
             stream at the plan seed+19). Outputs the digests/counts the scripts write to resume_*.pt/report.json.
  runtime    stage-8 runtime check (preflight.py _runtime: two hidden.detach(), 12 Dropout(p=.03), state hashes).
  extract    unpack a package ZIP into a runtime directory (later duplicate members overwrite, as setup_lcsd*.py did).
  compare    member-by-member comparison of two package ZIPs (bytes; for differing weight members also the
             state_dict equality and the differing non-state_dict payload keys).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXCLUDED_REAL, ROOT, read_json, sd_hash, write_json  # noqa: E402
from common_8_12 import WEIGHTS, base_steps  # noqa: E402

sys.path.insert(0, str(ROOT / 'src'))


def inventory(train_cases, sim_excluded):
    from realpde.data import discover_cases
    cases = [c for c in discover_cases(str(ROOT / 'data/extracted/train_real'), exclude_files=[EXCLUDED_REAL])
             if c.path.name in set(train_cases)]
    assert [c.path.name for c in cases] == list(train_cases), 'discover order / membership differs from the case list'
    assert all((c.height, c.width) == (64, 128) and not c.has_pressure for c in cases)   # released_lattice_dataset.py:8
    windows = sum(c.frames - 40 + 1 for c in cases)
    # sim_replay_all.py:17 excludes 7575_0.h5 by name in train_sim too; the fold rule then drops held regimes
    sims = [c for c in discover_cases(ROOT / 'data/extracted/train_sim', exclude_files=['7575_0.h5'])
            if c.path.name not in set(sim_excluded)]
    return {'real': [[c.path.name, c.frames] for c in cases], 'windows': windows, 'base_steps': base_steps(windows),
            'sim': [[c.path.name, c.frames] for c in sims], 'sim_windows': sum(c.frames - 40 + 1 for c in sims),
            'sim_excluded': sorted(sim_excluded)}


def _windows(frames):
    return [(i, s) for i, (_, f) in enumerate(frames) for s in range(f - 40 + 1)]


def replay_latcadf(frames, training, middle):
    import torch
    windows, lengths = _windows(frames), [f for _, f in frames]
    n, seed, steps, p = len(windows), training['seed'], training['steps'], float(training['cadence2_probability'])
    rng, aug = torch.Generator().manual_seed(seed + 1), torch.Generator().manual_seed(seed + 2)
    ph, cad = torch.Generator().manual_seed(seed + 3), torch.Generator().manual_seed(seed + 4)
    digest, adigest, pdigest, cdigest = (hashlib.sha256() for _ in range(4))
    order, cursor, seen = torch.randperm(n, generator=rng), 0, set()
    phase_counts, case_phase_counts, cadence_counts, fallbacks, reflected, mid = [0] * 4, [[0] * 4 for _ in frames], [0, 0], 0, 0, None
    first = None
    for step in range(1, steps + 1):
        if cursor >= n:
            order, cursor = torch.randperm(n, generator=rng), 0
        ix = order[cursor:cursor + 8]
        cursor += len(ix)
        seen.update(ix.tolist())
        digest.update(ix.numpy().tobytes())
        phases = torch.randint(4, (len(ix),), generator=ph)
        pdigest.update(phases.numpy().tobytes())
        for index, phase in zip(ix.tolist(), phases.tolist()):
            phase_counts[phase] += 1
            case_phase_counts[windows[index][0]][phase] += 1
        cads = (torch.rand(len(ix), generator=cad) < p).long()
        cdigest.update(cads.numpy().tobytes())
        requested = cads.tolist()
        actual = [c if (c == 0 or windows[i][1] + 79 <= lengths[windows[i][0]]) else 0 for i, c in zip(ix.tolist(), requested)]
        fallbacks += sum(1 for r, a in zip(requested, actual) if r == 1 and a == 0)
        for c in actual:
            cadence_counts[c] += 1
        flip = torch.rand(len(ix), generator=aug) < .5
        adigest.update(flip.numpy().tobytes())
        reflected += int(flip.sum())
        if step == 1:
            first = {'first_source_indices': ix.tolist(), 'first_reflection_mask': flip.tolist(), 'first_phases': phases.tolist()}
        if step == middle:
            mid = {'step': step, 'batch_digest': digest.hexdigest(), 'augmentation_digest': adigest.hexdigest(),
                   'phase_digest': pdigest.hexdigest()}
    assert len(seen) == n and all(min(v) > 0 for v in case_phase_counts)       # train.py:121-122
    return {'kind': 'latcadf', 'steps': steps, 'windows': n, 'batch_digest': digest.hexdigest(),
            'augmentation_digest': adigest.hexdigest(), 'phase_digest': pdigest.hexdigest(),
            'cadence_digest': cdigest.hexdigest(), 'phase_counts': phase_counts, 'case_phase_counts': case_phase_counts,
            'cadence_counts': cadence_counts, 'cadence_trailing_fallbacks': fallbacks, 'reflected_samples': reflected,
            'source_samples': sum(phase_counts), 'unique_sampled_windows': len(seen), 'middle': mid, **first}


def replay_sd10(frames, training, middle):
    import torch
    assert training['warmup_steps'] == 0, 'the recipe has no distillation warm-up phase'
    windows, lengths = _windows(frames), [f for _, f in frames]
    n, steps, p = len(windows), training['finetune_steps'], float(training['cadence2_probability'])
    lp = list(training.get('lattice_phases', [0, 1, 2, 3]))
    seed = training['seed'] + int(training.get('stream_seed_offset', 0))          # train_cycle4.py:138
    rng, ag = torch.Generator().manual_seed(seed + 1), torch.Generator().manual_seed(seed + 2)
    prng, crng = torch.Generator().manual_seed(seed + 3), torch.Generator().manual_seed(seed + 4)
    order, cursor, seen, digest, adigest = torch.randperm(n, generator=rng), 0, set(), hashlib.sha256(), hashlib.sha256()
    lat, cadc, fb, samples, reflected, mid = [0] * 4, [0, 0], 0, 0, 0, None
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
        fb += sum(1 for r, a in zip(req, cads) if r == 1 and a == 0)
        for q in phs:
            lat[q] += 1
        for c in cads:
            cadc[c] += 1
        flip = torch.rand(len(ix), generator=ag) < .5
        adigest.update(flip.numpy().tobytes())
        samples += len(ix)
        reflected += int(flip.sum())
        if step == middle:
            mid = {'step': step, 'batch_digest': digest.hexdigest(), 'augmentation_digest': adigest.hexdigest()}
    assert len(seen) == n                                                          # train_sd10.py:183
    return {'kind': 'sd10', 'steps': steps, 'windows': n, 'stream_seed': seed, 'batch_digest': digest.hexdigest(),
            'augmentation_digest': adigest.hexdigest(), 'lattice_phase_counts': lat, 'cadence_counts': cadc,
            'cadence_trailing_fallbacks': fb, 'source_samples': samples, 'reflected_samples': reflected,
            'unique_sampled_windows': len(seen), 'middle': mid}


def replay_sim(sim_frames, training, calls):
    import torch
    windows = _windows(sim_frames)
    n = len(windows)
    rng = torch.Generator().manual_seed(training['seed'] + 19)                    # sim_replay_all.py:13,33 (no offset)
    order, cursor, seen, digest, samples, first = torch.randperm(n, generator=rng), 0, set(), hashlib.sha256(), 0, None
    for call in range(calls):
        if cursor >= n:
            order, cursor = torch.randperm(n, generator=rng), 0
        idx = order[cursor:cursor + 8]
        cursor += len(idx)
        seen.update(idx.tolist())
        digest.update(idx.numpy().tobytes())
        samples += len(idx)
        if call == 0:
            first = [[sim_frames[windows[i][0]][0], windows[i][1]] for i in idx.tolist()]
    return {'policy_cases': len(sim_frames), 'windows': n, 'calls': calls, 'samples': samples,
            'unique_windows': len(seen), 'digest': digest.hexdigest(), 'first_sample_ids': first}


def replay(plan, inv, kind):
    tr = plan['training']
    out = replay_latcadf(inv['real'], tr, plan['middle_step']) if kind == 'latcadf' else replay_sd10(inv['real'], tr, plan['middle_step'])
    assert out['windows'] == plan['sampling']['windows'], 'replayed window count differs from the plan'
    if kind == 'sd10' and tr.get('sim_replay'):
        out['simulation_replay'] = replay_sim(inv['sim'], tr, tr['finetune_steps'])
    return out


def runtime_check(plan):
    import preflight
    _, _, initial, arch = preflight._runtime(plan)                               # train.py:42-51 on CPU
    assert arch['dropout_modules'] == 12 and arch['dropout_p'] == [0.03], arch       # train.py:50
    return {'status': 'RUNTIME_OK', 'initial_states': initial, 'architecture': arch}


def extract(archive, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for info in z.infolist():
            if info.is_dir():
                continue
            target = out / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    return {'archive': str(Path(archive).resolve().relative_to(ROOT)), 'members': names,
            'duplicates': sorted({n for n in names if names.count(n) > 1})}


def _payload(data):
    import torch
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)


def _strip(value):
    import torch
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip(v) for v in value]
    return 'tensor' + str(tuple(value.shape)) if torch.is_tensor(value) else value


def compare(rebuilt, reference):
    """Per-member byte identity of `rebuilt` against `reference` (both ZIPs)."""
    import torch
    with zipfile.ZipFile(rebuilt) as a, zipfile.ZipFile(reference) as b:
        an, bn = a.namelist(), b.namelist()
        rows = []
        for name in dict.fromkeys(an + bn):
            if name not in an or name not in bn:
                rows.append({'member': name, 'status': 'only_rebuilt' if name in an else 'only_reference'})
                continue
            da, db = a.read(name), b.read(name)
            row = {'member': name, 'sha256_rebuilt': hashlib.sha256(da).hexdigest(),
                   'sha256_reference': hashlib.sha256(db).hexdigest()}
            row['status'] = 'identical' if da == db else 'different'
            if da != db and name in WEIGHTS:
                pa, pb = _payload(da), _payload(db)
                sa, sb = pa['state_dict'], pb['state_dict']
                row['state_dict_equal'] = list(sa) == list(sb) and all(torch.equal(v, sb[k]) for k, v in sa.items())
                row['state_hash_rebuilt'], row['state_hash_reference'] = sd_hash(sa), sd_hash(sb)
                ka = {k: _strip(v) for k, v in pa.items() if k != 'state_dict'}
                kb = {k: _strip(v) for k, v in pb.items() if k != 'state_dict'}
                row['payload_keys_only_reference'] = sorted(set(kb) - set(ka))
                row['payload_keys_only_rebuilt'] = sorted(set(ka) - set(kb))
                row['payload_keys_differing'] = sorted(k for k in set(ka) & set(kb) if ka[k] != kb[k])
            rows.append(row)
        only_ref = {hashlib.sha256(b.read(r['member'])).hexdigest(): r['member'] for r in rows if r['status'] == 'only_reference'}
        for row in rows:                  # a member present under another name with identical bytes
            if row['status'] == 'only_rebuilt':
                twin = only_ref.get(hashlib.sha256(a.read(row['member'])).hexdigest())
                if twin:
                    row.update(status='renamed_identical', reference_member=twin)
        order_equal = [n for n in an if n in bn] == [n for n in bn if n in an]
    return {'rebuilt': str(Path(rebuilt).resolve().relative_to(ROOT)), 'reference': str(Path(reference).resolve().relative_to(ROOT)),
            'members_rebuilt': len(an), 'members_reference': len(bn), 'common_member_order_equal': order_equal,
            'identical': sum(r['status'] == 'identical' for r in rows), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory')
    p.add_argument('--cases-json', type=Path, required=True)
    p.add_argument('--sim-excluded-json', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('replay')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--kind', choices=['latcadf', 'sd10'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('runtime')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('extract')
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p = sub.add_parser('compare')
    p.add_argument('--rebuilt', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if args.command == 'inventory':
        result = inventory(read_json(args.cases_json), read_json(args.sim_excluded_json))
    elif args.command == 'replay':
        result = replay(read_json(args.plan), read_json(args.inventory), args.kind)
    elif args.command == 'runtime':
        result = runtime_check(read_json(args.plan))
    elif args.command == 'extract':
        result = extract(args.zip, args.out)
        args.out = args.record
    else:
        result = compare(args.rebuilt, args.reference)
    write_json(args.out, result)
    print(json.dumps({'status': 'OK', 'command': args.command, 'out': str(args.out)}))


if __name__ == '__main__':
    main()
