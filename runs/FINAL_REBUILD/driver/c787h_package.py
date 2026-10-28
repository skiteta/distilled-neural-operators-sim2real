"""Stage 7 packaging (C787H): S6 EMA point bytes + S7 EMA heads -> LATCADF-consumable runtime/.

Lifted from runs/C787R_full81_point_preserved/package.py:5,10-14,23,27 (spec Stage 7).
The one change: no template ZIP (package.py:7-9,17-21). model.pt is the byte copy of the
S6 ema/model.pt, checked with torch.equal against the resume checkpoint's ema_point.
Hard-coded 9232/9233 become plan['resume_step'] (identical for the full release).
Evaluation pins (package.py:29-35) are not part of the rebuild. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COMPONENT_FILES, HEAD_PREFIXES, ROOT, RUNTIME_CODE, read_json, rel,  # noqa: E402
                    sd_hash, sha256, write_json)

ZIP_LIMIT = 256_000_000


def _deterministic_zip(target, folder, names):
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (folder / name).read_bytes())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and sum(i.file_size for i in z.infolist()) < ZIP_LIMIT
        assert sorted(z.namelist()) == sorted(names)


def c787h_package(rt5, stage_dir, out_dir=None, make_zip=True):
    """rt5: S5 common_runtime; stage_dir: S7 dir holding candidate_plan.json + training_candidate/.
    Writes out_dir/{runtime/, deployed_checkpoint.pt, package_audit.json[, C787H_EMA<end>_REBUILT.zip]}."""
    import torch
    torch.set_num_threads(2)
    rt5, stage_dir = Path(rt5).resolve(strict=True), Path(stage_dir).resolve(strict=True)
    out_dir = Path(out_dir).resolve() if out_dir else stage_dir
    plan_path = stage_dir / 'candidate_plan.json'
    plan = read_json(plan_path)
    t = stage_dir / 'training_candidate'
    r = read_json(t / 'report.json')
    end, resume_step = plan['training']['steps'], plan['resume_step']
    path = t / f'resume_{end}.pt'
    resume = ROOT / plan['resume_checkpoint']
    state = torch.load(path, map_location='cpu', weights_only=False)
    old = torch.load(resume, map_location='cpu', weights_only=False)
    # package.py:5 (self-checks against the stage's own plan and report)
    assert r['status'] == 'COMPLETE'
    assert r['steps_actual'] == state['step'] == state['scheduler']['last_epoch'] == end
    assert state['plan_sha256'] == sha256(plan_path)
    assert {int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v} == {end}
    assert r['new_updates_actual'] == end - resume_step and old['step'] == resume_step
    # package.py:10-14: independent replay of the continuation-suffix sampler.
    g = torch.Generator()
    g.set_state(old['rng'])
    order, cursor, digest, n = old['sampler_order'], old['sampler_cursor'], hashlib.sha256(), plan['sampling']['windows']
    for _ in range(resume_step + 1, end + 1):
        if cursor >= n:
            order = torch.randperm(n, generator=g)
            cursor = 0
        ix = order[cursor:cursor + 8]
        cursor += len(ix)
        digest.update(ix.numpy().tobytes())
    assert state['batch_digest'] == digest.hexdigest() and state['prior_batch_digest'] == old['batch_digest']
    assert cursor == state['sampler_cursor'] and torch.equal(order, state['sampler_order'])
    assert torch.equal(g.get_state(), state['rng'])
    assert sha256(resume) == plan['resume_checkpoint_sha256']      # chain-internal (computed at plan time)
    # Replaces package.py:7-9/19-21: the point is the S6 EMA export itself, state-checked.
    ema_path = ROOT / plan['baseline_training'] / 'ema' / 'model.pt'
    ema_model = torch.load(ema_path, map_location='cpu', weights_only=False)
    assert list(ema_model['state_dict']) == list(old['ema_point'])
    assert all(torch.equal(v, old['ema_point'][k]) for k, v in ema_model['state_dict'].items())

    runtime = out_dir / 'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    for name, expected in RUNTIME_CODE.items():                    # package.py:24 code pin
        assert sha256(rt5 / name) == expected, name
        (runtime / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rt5 / name, runtime / name)
    shutil.copyfile(ema_path, runtime / 'model.pt')
    heads_sha = sha256(path)
    for name, prefix in HEAD_PREFIXES.items():                      # package.py:23
        payload = torch.load(rt5 / name, map_location='cpu', weights_only=False)
        payload['state_dict'] = {k[len(prefix):]: v for k, v in state['ema_heads'].items() if k.startswith(prefix)}
        payload['schedule_continuation'] = {'resume_step': resume_step, 'step': end, 'checkpoint_sha256': heads_sha}
        torch.save(payload, runtime / name)
    names = sorted(str(p.relative_to(runtime)) for p in runtime.rglob('*') if p.is_file())
    assert set(names) == set(RUNTIME_CODE) | set(COMPONENT_FILES) and len(names) == 15
    members = {name: sha256(runtime / name) for name in names}
    assert members['model.pt'] == sha256(ema_path)                  # point bytes preserved
    states = {}
    for name in COMPONENT_FILES:                                    # reload check of all five deployed states
        sd = torch.load(runtime / name, map_location='cpu', weights_only=False)['state_dict']
        ref = old['ema_point'] if name == 'model.pt' else {k[len(HEAD_PREFIXES[name]):]: v for k, v in state['ema_heads'].items() if k.startswith(HEAD_PREFIXES[name])}
        assert list(sd) == list(ref) and all(torch.equal(v, ref[k]) for k, v in sd.items()), name
        states[name] = sd_hash(sd)
    deployed = out_dir / 'deployed_checkpoint.pt'                   # package.py:27
    torch.save(dict(ema_point=old['ema_point'], ema_heads=state['ema_heads'], point_source=plan['resume_checkpoint'],
                    heads_source=rel(path), heads_source_sha256=heads_sha, step=end), deployed)
    archive = None
    if make_zip:
        archive = out_dir / f'C787H_EMA{end}_REBUILT.zip'
        _deterministic_zip(archive, runtime, names)
    audit = {'status': 'COMPLETE', 'runtime': rel(runtime), 'runtime_members': members, 'deployed_states': states,
             'resume_step': resume_step, 'step': end, 'heads_source': rel(path), 'heads_source_sha256': heads_sha,
             'point_source': plan['resume_checkpoint'], 'point_bytes_source': rel(ema_path),
             'deployed_checkpoint': rel(deployed), 'deployed_checkpoint_sha256': sha256(deployed),
             'archive': None if archive is None else rel(archive),
             'archive_sha256': None if archive is None else sha256(archive),
             'all_five_deployed_states_exact': True, 'original_point_bytes_preserved': True,
             'suffix_sampler_independently_replayed': True, 'suffix_batch_digest': state['batch_digest'],
             'weight_hashes': 'recorded only; no historical template or byte pin'}
    write_json(out_dir / 'package_audit.json', audit)
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True, help='S5 common_runtime')
    parser.add_argument('--stage-dir', type=Path, required=True, help='S7 dir (candidate_plan.json, training_candidate/)')
    parser.add_argument('--out', type=Path, help='output dir (default: --stage-dir)')
    parser.add_argument('--no-zip', action='store_true')
    args = parser.parse_args(argv)
    runtime = c787h_package(args.runtime, args.stage_dir, args.out, make_zip=not args.no_zip)
    audit = read_json(runtime.parent / 'package_audit.json')
    print(json.dumps({'status': 'PACKAGED', 'runtime': rel(runtime), 'members': len(audit['runtime_members']),
                      'step': audit['step']}))


if __name__ == '__main__':
    main()
