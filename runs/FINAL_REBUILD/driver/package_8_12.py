"""Stage 8-13 packaging (CPU): midpoint/final EMA 50:50 -> runtime/ + deterministic ZIP + deployed checkpoint.

Lifted from the historical packagers; the arithmetic, payload handling, member order and recipe members are
kept line for line, the template sources are changed (B2):
  latcadf  runs/LATCAD_upperRe/package_latcad.py:7-22 (build) + runs/LATCADF_full81/finalize_latcadf.py:22-30
           (recipe + deployed checkpoint). Historical: weight payload templates from the C1017H ZIP (a non-lineage
           weight-bearing ZIP) and code members from the C973H ZIP. Rebuild: payload templates from the plan's
           teacher_archive = the stage-7 (C787H-equivalent) package, code from the vendored runtime_code_c973h/.
  sd10     runs/SD10/finalize_sd10.py:11-54 (package), as called by runs/CYCLE4/finish_generic.py:10-11
           (+ its duplicate-member assert). Historical: payload templates + carried training_recipe/ from the
           teacher ZIP, code from the C973H ZIP. Rebuild: teacher ZIP = the previous stage's rebuilt package,
           code from runtime_code_c973h/. training_recipe/<CAND>_train.py.txt is the script named in the stage's
           training_script.txt (B5: the historical LC4BH ZIP carries train_sd10.py under that name).
The evaluation, qualification and timing steps of finish_generic.py / finalize_*.py are not part of the rebuild.
New chain checks: resume checkpoints carry the sha256 of the plan the training ran with (`--trained-plan`,
default --plan). Output: OUT/{runtime/, <CAND>_REBUILT.zip, <CAND>_deployed_checkpoint.pt, package_audit.json}.
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
from common import ROOT, rel, read_json, sd_hash, sha256, write_json  # noqa: E402
from common_8_12 import (CODE_PINS_8_12, LIFTED_FROM, RUNTIME_C973, RUNTIME_C973_DIR, RUNTIME_C973_ORDER,  # noqa: E402
                         T8, WEIGHTS)

ZIP_LIMIT = 256_000_000
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _average(a, b):
    """package_latcad.py:7-8 == finalize_sd10.py:17 (a = midpoint, b = final EMA)."""
    assert a.keys() == b.keys()
    return {k: (a[k] * .5 + b[k] * .5 if a[k].is_floating_point() or a[k].is_complex() else a[k].clone()) for k in a}


def _split(point, heads):
    return {name: (point if key == 'point' else {k[len(key) + 1:]: v for k, v in heads.items() if k.startswith(key + '.')})
            for name, key in WEIGHTS.items()}


def _runtime_code(plan):
    code = {}
    for name, digest in RUNTIME_C973.items():
        data = (RUNTIME_C973_DIR / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit('vendored runtime code changed: ' + name)
        code[name] = data
    assert set(code) | set(WEIGHTS) == set(RUNTIME_C973_ORDER)
    baseline = plan.get('runtime_baseline') or {}
    recorded = baseline.get('members') or baseline.get('runtime_members') or {}
    bad = [n for n, h in recorded.items() if n not in WEIGHTS and RUNTIME_C973.get(n) != h]
    if bad:                                                  # replaces finalize_sd10.py:21/:50 (C973H ZIP sha and bytes)
        raise SystemExit('plan runtime_baseline members differ from the vendored runtime code: ' + ', '.join(bad))
    return code


def _payload_bytes(template, state, model_config=None):
    import torch
    payload = torch.load(io.BytesIO(template), map_location='cpu', weights_only=False)
    payload['state_dict'] = state
    if model_config is not None:
        payload['model_config'] = model_config
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _archive(plan, override):
    ref = plan['teacher_archive']
    path = Path(override) if override else ROOT / ref['path']
    path = (path if path.is_absolute() else ROOT / path).resolve(strict=True)
    if sha256(path) != ref['sha256']:                         # finalize_sd10.py:21 (same rule for the stage-8 template)
        raise SystemExit(f'template archive hash differs from the plan record: {rel(path)}')
    return path


def _load_training(plan, trained_plan, training_dir):
    import torch
    t = Path(training_dir)
    steps, middle = plan['training']['steps'], plan['middle_step']
    end = torch.load(t / f'resume_{steps}.pt', map_location='cpu', weights_only=False)
    mid = torch.load(t / f'resume_{middle}.pt', map_location='cpu', weights_only=False)
    assert end['plan_sha256'] == mid['plan_sha256'] == sha256(trained_plan), 'resume checkpoints belong to another plan'
    assert end['step'] == steps and mid['step'] == middle
    return end, mid


def _write(out, candidate, entries, states, deployed, extra):
    names = [n for n, _ in entries]
    assert len(names) == len(set(names)), 'duplicate ZIP member names'        # finish_generic.py:11 / finalize_sd10.py:47
    runtime = out / 'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    for name, data in entries:
        (runtime / name).parent.mkdir(parents=True, exist_ok=True)
        (runtime / name).write_bytes(data)
    archive = out / f'{candidate}_REBUILT.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.namelist() == names and sum(i.file_size for i in z.infolist()) < ZIP_LIMIT
        assert all(z.read(n) == d for n, d in entries)
    import torch
    for name in WEIGHTS:                                     # finalize_sd10.py:51-53 reload check of all five states
        sd = torch.load(runtime / name, map_location='cpu', weights_only=False)['state_dict']
        ref = states[name]
        assert list(sd) == list(ref) and all(torch.equal(v, ref[k]) for k, v in sd.items()), name
    audit = {'status': 'COMPLETE', 'candidate': candidate, 'runtime': rel(runtime),
             'members': [[n, hashlib.sha256(d).hexdigest()] for n, d in entries],
             'deployed_states': {n: sd_hash(states[n]) for n in WEIGHTS},
             'deployed_checkpoint': rel(deployed), 'deployed_checkpoint_sha256': sha256(deployed),
             'archive': rel(archive), 'archive_sha256': sha256(archive),
             'runtime_code': {'dir': rel(RUNTIME_C973_DIR), 'sha256': RUNTIME_C973},
             'lifted_from': LIFTED_FROM, 'weight_hashes': 'recorded only; no historical template or byte pin', **extra}
    write_json(out / 'package_audit.json', audit)
    return audit


def package_latcadf(plan_path, out, trained_plan=None, training_dir=None, template=None):
    import torch
    torch.set_num_threads(2)
    plan_path = Path(plan_path).resolve(strict=True)
    trained_plan = Path(trained_plan).resolve(strict=True) if trained_plan else plan_path
    m = read_json(plan_path)
    t = Path(training_dir).resolve(strict=True) if training_dir else plan_path.parent / 'training_candidate'
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = read_json(t / 'report.json')
    assert rep['status'] == 'COMPLETE' and rep['steps_actual'] == m['training']['steps']        # finalize_latcadf.py:22
    end, mid = _load_training(m, trained_plan, t)                                              # package_latcad.py:10-11
    states = _split(_average(mid['ema_point'], end['ema_point']), _average(mid['ema_heads'], end['ema_heads']))  # :12-13
    tr = m['training']
    recipe = {'training_recipe/released_lattice_cadence.json': json.dumps(dict(                  # finalize_latcadf.py:23-27
        lattice_phases=tr['lattice_phases'], cadence2_probability=tr['cadence2_probability'],
        cadence_generator='independent CPU generator seed+4', trailing_starts='spacing 1 when fewer than 79 frames remain',
        cadence_counts=rep['cadence_counts'], cadence_trailing_fallbacks=rep['cadence_trailing_fallbacks'],
        parent='C1099H lattice-only full81 plan', endpoint='fixed midpoint/final EMA 50:50'), indent=2).encode(),
        'training_recipe/latcad_train.py.txt': (ROOT / T8).read_bytes()}
    assert sha256(ROOT / T8) == CODE_PINS_8_12[T8]
    code, archive = _runtime_code(m), _archive(m, template)
    entries = []
    with zipfile.ZipFile(archive) as origin:                                                  # package_latcad.py:15-22
        for name in RUNTIME_C973_ORDER:
            entries.append((name, _payload_bytes(origin.read(name), states[name]) if name in WEIGHTS else code[name]))
    for name, data in recipe.items():
        assert name.startswith('training_recipe/') and name not in RUNTIME_C973_ORDER
        entries.append((name, data))
    deployed = out / 'candidate_deployed_checkpoint.pt'     # finalize_latcadf.py:29-30 (torch.save embeds the file stem)
    torch.save(dict(ema_point=states['model.pt'], ema_heads={WEIGHTS[n] + '.' + k: v for n, s in states.items()
                                                             if n != 'model.pt' for k, v in s.items()}), deployed)
    return _write(out, m['candidate'], entries, states, deployed, {
        'kind': 'latcadf', 'plan': rel(plan_path), 'plan_sha256': sha256(plan_path), 'trained_plan_sha256': sha256(trained_plan),
        'training_dir': rel(t), 'steps': m['training']['steps'], 'middle_step': m['middle_step'],
        'template_archive': {'path': rel(archive), 'sha256': sha256(archive), 'role': 'weight payload templates (non-state_dict keys)'},
        'candidate_recipe': sorted(recipe), 'endpoint': 'midpoint/final EMA 50:50'})


def package_sd10(plan_path, out, trained_plan=None, training_dir=None, teacher=None, script_spec=None):
    import torch
    torch.set_num_threads(2)
    plan_path = Path(plan_path).resolve(strict=True)
    trained_plan = Path(trained_plan).resolve(strict=True) if trained_plan else plan_path
    m = read_json(plan_path)
    t = Path(training_dir).resolve(strict=True) if training_dir else plan_path.parent / 'training_candidate'
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep, cand = read_json(t / 'report.json'), m['candidate']
    depth = int(m['training'].get('student_depth', 10))
    if depth != 10:
        raise SystemExit('depth != 10 needs depth_runtime(submission.py) (finalize_sd10.py:27-29); not part of this lineage')
    assert rep['status'] == 'COMPLETE' and rep['teacher_unchanged'] and rep['student_depth'] == depth and (   # :14
        rep['fine_source_and_reflection_stream_exact_to_teacher_fit'] or not m['training'].get('replay_teacher_stream', True))
    end, mid = _load_training(m, trained_plan, t)                                              # :15-16
    assert end['model_config']['depth'] == depth
    point, heads = _average(mid['ema_point'], end['ema_point']), _average(mid['ema_heads'], end['ema_heads'])   # :17-18
    deployed = out / f'{cand}_deployed_checkpoint.pt'                                          # :19
    torch.save(dict(ema_point=point, ema_heads=heads, model_config=end['model_config'], step=end['step']), deployed)
    states = _split(point, heads)
    code, archive = _runtime_code(m), _archive(m, teacher)
    spec_file = Path(script_spec) if script_spec else plan_path.parent / 'training_script.txt'  # :41
    spec = [x.strip() for x in spec_file.read_text().splitlines() if x.strip()]
    for x in spec:
        if CODE_PINS_8_12.get(x) != sha256(ROOT / x):
            raise SystemExit('training script named in training_script.txt is not a pinned stage 8-13 script: ' + x)
    entries, written, carried = [], set(), []
    with zipfile.ZipFile(archive) as old:                                                     # :22-38
        for name in RUNTIME_C973_ORDER:
            if name.startswith('training_recipe/'):
                continue
            if name in WEIGHTS:
                data = _payload_bytes(old.read(name), states[name], end['model_config'] if WEIGHTS[name] == 'point' else None)
            else:
                data = code[name]
            entries.append((name, data))
            written.add(name)
        for name in old.namelist():
            if name.startswith('training_recipe/') and name not in written:
                entries.append((name, old.read(name)))
                written.add(name)
                carried.append(name)
    recipe = [(f'training_recipe/{cand}_continuation.json', json.dumps(m, ensure_ascii=False, indent=2).encode()),   # :42-45
              (f'training_recipe/{cand}_train.py.txt', (ROOT / spec[0]).read_bytes())]
    recipe += [(f'training_recipe/{cand}_{Path(x).name}.txt', (ROOT / x).read_bytes()) for x in spec[1:]]
    for name, data in recipe:
        assert name not in written, name
        entries.append((name, data))
        written.add(name)
    return _write(out, cand, entries, states, deployed, {
        'kind': 'sd10', 'plan': rel(plan_path), 'plan_sha256': sha256(plan_path), 'trained_plan_sha256': sha256(trained_plan),
        'training_dir': rel(t), 'steps': m['training']['steps'], 'middle_step': m['middle_step'],
        'template_archive': {'path': rel(archive), 'sha256': sha256(archive), 'role': 'teacher package: weight payload templates + carried training_recipe/'},
        'training_script_spec': spec, 'carried_recipe': carried, 'candidate_recipe': [n for n, _ in recipe],
        'endpoint': 'midpoint/final finetuning EMA 50:50'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['latcadf', 'sd10'])
    parser.add_argument('--plan', type=Path, required=True, help='plan driving the packaging (candidate, steps, archives)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trained-plan', type=Path, help='plan the training ran with (resume plan_sha256); default --plan')
    parser.add_argument('--training-dir', type=Path, help='default: <plan dir>/training_candidate')
    parser.add_argument('--template-zip', type=Path, help="override of plan['teacher_archive']['path'] (sha256 still checked)")
    parser.add_argument('--script-spec', type=Path, help='training_script.txt (sd10; default: <plan dir>/training_script.txt)')
    args = parser.parse_args(argv)
    if args.kind == 'latcadf':
        audit = package_latcadf(args.plan, args.out, args.trained_plan, args.training_dir, args.template_zip)
    else:
        audit = package_sd10(args.plan, args.out, args.trained_plan, args.training_dir, args.template_zip, args.script_spec)
    print(json.dumps({'status': 'PACKAGED', 'candidate': audit['candidate'], 'members': len(audit['members']),
                      'archive_sha256': audit['archive_sha256']}))


if __name__ == '__main__':
    main()
