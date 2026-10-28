"""Stage 13-15 packaging (CPU) and the deployable final ZIP.

  stage  runs/SD10/finalize_sd10.py:18-64 package() (as called by runs/CYCLE4/finish_generic.py:10-11) for any student
         depth: midpoint/final fine-tune EMA 50:50 -> runtime/ + <CAND>_REBUILT.zip + <CAND>_deployed_checkpoint.pt +
         package_audit.json. Reuses package_8_12's lifted helpers (averaging, payload templates, vendored runtime code,
         training-plan checks, deterministic writer with the reload check). Depth != 10: submission.py =
         depth_runtime(vendored submission.py) (finalize_sd10.py:35-37); depth_runtime is compiled from the pinned
         scripts/depth_distillation_cycle_v2.py (only that function; the module and its imports are never loaded).
         B2 as in stages 8-12: weight-payload templates + carried training_recipe/ come from the previous rebuilt package
         (plan teacher_archive), executable runtime code from runtime_code_c973h/ instead of the weight-bearing C973H ZIP.
  final  runs/EXACTLANE/make_fixed5.py:23-40 (F5: network batch 3 -> 5, predict log line, fixed_transfer install; adds
         the member fixed_transfer.py) then runs/EXACTLANE/make_malloc.py:39-54 (MA: glibc heap block after the
         __future__ import); the literal strings are read (AST) from the pinned scripts, the member checks of both scripts
         and of runs/TTAKD/stage_ready_d4.py:32-51 are restated against the vendored runtime code (instead of the
         historical SIMAHF5 ZIP). Output: OUT/<C>F5_FIXED5_REBUILT.zip, OUT/<C>F5MA_FIXED5_REBUILT.zip (the deliverable),
         OUT/runtime/ (its members), OUT/final_audit.json. ZIP entries are written deterministically (fixed timestamp).
The evaluation, qualification, timing and READY staging steps of finish_generic.py / make_fixed5.py / make_malloc.py /
stage_d4.sh select and verify candidates; they do not change the artifact and are not part of the rebuild.
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import sys

sys.dont_write_bytecode = True   # never write __pycache__ into the project tree
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_json, rel, sd_hash, sha256, write_json  # noqa: E402
from common_8_12 import CODE_PINS_8_12, RUNTIME_C973_DIR, RUNTIME_C973_ORDER, WEIGHTS  # noqa: E402
from common_13_15 import (CODE_PINS_13_15, DEPTH_SOURCE, FIXED_TRANSFER, LIFTED_FROM_13_15, PACKAGING_PINS,  # noqa: E402
                          READY_CODE_MEMBERS, f5_constants, lifted_function, ma_constants, sha256_bytes)
from package_8_12 import FIXED_TIME, ZIP_LIMIT, _archive, _average, _load_training, _payload_bytes, _runtime_code, _split, _write  # noqa: E402


def depth_runtime():
    return lifted_function(DEPTH_SOURCE, PACKAGING_PINS[DEPTH_SOURCE], 'depth_runtime')


def package_stage(plan_path, out, trained_plan=None, training_dir=None, teacher=None, script_spec=None):
    import torch
    torch.set_num_threads(2)
    plan_path = Path(plan_path).resolve(strict=True)
    trained_plan = Path(trained_plan).resolve(strict=True) if trained_plan else plan_path
    m = read_json(plan_path)
    t = Path(training_dir).resolve(strict=True) if training_dir else plan_path.parent / 'training_candidate'
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep, cand, tr = read_json(t / 'report.json'), m['candidate'], m['training']
    depth = int(tr.get('student_depth', 10))
    if tr.get('no_multiscale'):
        raise SystemExit('no_multiscale (finalize_sd10.py:38-39) is not part of this lineage')
    assert rep['status'] == 'COMPLETE' and rep['teacher_unchanged'] and rep['student_depth'] == depth and (   # :22
        rep['fine_source_and_reflection_stream_exact_to_teacher_fit'] or not tr.get('replay_teacher_stream', True))
    end, mid = _load_training(m, trained_plan, t)                                             # :23-24
    assert end['model_config']['depth'] == depth
    point, heads = _average(mid['ema_point'], end['ema_point']), _average(mid['ema_heads'], end['ema_heads'])   # :25-26
    deployed = out / f'{cand}_deployed_checkpoint.pt'                                         # :27
    torch.save(dict(ema_point=point, ema_heads=heads, model_config=end['model_config'], step=end['step']), deployed)
    states = _split(point, heads)
    code, archive = _runtime_code(m), _archive(m, teacher)
    if depth != 10:                                                                            # :35-37
        code['submission.py'] = depth_runtime()(code['submission.py'])
    spec_file = Path(script_spec) if script_spec else plan_path.parent / 'training_script.txt'  # :51
    spec = [x.strip() for x in spec_file.read_text().splitlines() if x.strip()]
    pins = {**CODE_PINS_8_12, **CODE_PINS_13_15}
    for x in spec:
        if pins.get(x) != sha256(ROOT / x):
            raise SystemExit('training script named in training_script.txt is not a pinned stage 8-15 script: ' + x)
    entries, written, carried = [], set(), []
    with zipfile.ZipFile(archive) as old:                                                     # :30-48
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
    recipe = [(f'training_recipe/{cand}_continuation.json', json.dumps(m, ensure_ascii=False, indent=2).encode()),   # :52-55
              (f'training_recipe/{cand}_train.py.txt', (ROOT / spec[0]).read_bytes())]
    recipe += [(f'training_recipe/{cand}_{Path(x).name}.txt', (ROOT / x).read_bytes()) for x in spec[1:]]
    for name, data in recipe:
        assert name not in written, name
        entries.append((name, data))
        written.add(name)
    vendored_sub = (RUNTIME_C973_DIR / 'submission.py').read_bytes()
    for name, data in entries:                                                                 # :58-60 runtime bytes unchanged
        if name in WEIGHTS or name.startswith('training_recipe/'):
            continue
        expected = depth_runtime()(vendored_sub) if (name == 'submission.py' and depth != 10) else (RUNTIME_C973_DIR / name).read_bytes()
        assert data == expected, name
    return _write(out, cand, entries, states, deployed, {
        'kind': 'stage_13_15', 'student_depth': depth, 'plan': rel(plan_path), 'plan_sha256': sha256(plan_path),
        'trained_plan_sha256': sha256(trained_plan), 'training_dir': rel(t), 'steps': tr['steps'], 'middle_step': m['middle_step'],
        'template_archive': {'path': rel(archive), 'sha256': sha256(archive),
                             'role': 'teacher package: weight payload templates + carried training_recipe/'},
        'training_script_spec': spec, 'carried_recipe': carried, 'candidate_recipe': [n for n, _ in recipe],
        'submission_py': {'sha256': sha256_bytes(code['submission.py']),
                          'transform': 'depth_runtime (finalize_sd10.py:35-37)' if depth != 10 else 'none'},
        'endpoint': 'midpoint/final finetuning EMA 50:50', 'lifted_from': LIFTED_FROM_13_15,
        'packaging_pins': {DEPTH_SOURCE: PACKAGING_PINS[DEPTH_SOURCE]} if depth != 10 else {}})


def _zip(path, entries):
    names = [n for n, _ in entries]
    assert len(names) == len(set(names)), 'duplicate ZIP member names'
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and z.namelist() == names and sum(i.file_size for i in z.infolist()) < ZIP_LIMIT
        assert all(z.read(n) == d for n, d in entries)


def _diff(a, b, sign):
    return [x for x in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='', n=0)
            if x[:1] == sign and not x.startswith(sign * 3)]


def f5_submission(data, c):
    """make_fixed5.py:26-34."""
    s = data.decode()
    assert s.count(c['old']) == 1 and s.count(c['async_install']) == 1
    s2 = s.replace(c['old'], c['new'])
    for a, b in c['pairs']:
        assert s2.count(a) == 1, a
        s2 = s2.replace(a, b)
    s2 = s2 + c['tail']
    new_lines = set(c['new'].splitlines())
    expected = ['-' + x for x in c['old'].splitlines() if x not in new_lines] + ['-' + a.rstrip('\n') for a, _ in c['pairs']]
    assert sorted(_diff(s, s2, '-')) == sorted(expected), _diff(s, s2, '-')
    return s2.encode()


def ma_submission(data, c):
    """make_malloc.py:45-49."""
    s = data.decode()
    assert s.count(c['anchor']) == 1
    s2 = s.replace(c['anchor'], c['anchor'] + c['block'])
    diff = [x for x in difflib.unified_diff(s.splitlines(), s2.splitlines(), lineterm='', n=0)
            if x[:1] in '+-' and not x.startswith(('+++', '---'))]
    assert all(x.startswith('+') for x in diff) and len(diff) == len(c['block'].splitlines()), diff
    return s2.encode()


def package_final(source_zip, out, candidate, suffix='MA'):
    import torch
    source_zip = Path(source_zip).resolve(strict=True)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    f5c, mac = f5_constants(), ma_constants(suffix)
    fixed = (ROOT / FIXED_TRANSFER).read_bytes()
    if sha256_bytes(fixed) != PACKAGING_PINS[FIXED_TRANSFER]:
        raise SystemExit('fixed_transfer.py differs from its pin')
    with zipfile.ZipFile(source_zip) as z:
        src = [(n, z.read(n)) for n in z.namelist()]
    names = [n for n, _ in src]
    assert len(names) == len(set(names)) and 'fixed_transfer.py' not in names
    # ---- F5 (make_fixed5.py:23-40)
    f5 = [(n, f5_submission(d, f5c) if n == 'submission.py' else d) for n, d in src] + [('fixed_transfer.py', fixed)]
    assert all(d == dict(src)[n] for n, d in f5 if n not in ('submission.py', 'fixed_transfer.py'))
    f5_zip = out / f'{candidate}F5_FIXED5_REBUILT.zip'
    _zip(f5_zip, f5)
    # ---- MA (make_malloc.py:42-54)
    ma = [(n, ma_submission(d, mac) if n == 'submission.py' else d) for n, d in f5]
    assert [n for n, _ in ma] == [n for n, _ in f5]
    assert all(d == dict(f5)[n] for n, d in ma if n != 'submission.py')
    final = f'{candidate}F5{suffix}'
    ma_zip = out / f'{final}_FIXED5_REBUILT.zip'
    _zip(ma_zip, ma)
    runtime = out / 'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    for name, data in ma:
        (runtime / name).parent.mkdir(parents=True, exist_ok=True)
        (runtime / name).write_bytes(data)
    # ---- stage_ready_d4.py:32-51, restated against the vendored runtime code (no historical ZIP read)
    payload = torch.load(io.BytesIO(dict(ma)['model.pt']), map_location='cpu', weights_only=False)
    depth = int(payload['model_config']['depth'])
    vendored = _runtime_code({})                      # vendored runtime code, every file checked against its pin
    base_sub = vendored['submission.py'] if depth == 10 else depth_runtime()(vendored['submission.py'])
    expected_sub = ma_submission(f5_submission(base_sub, f5c), mac)
    code = {n: d for n, d in ma if n not in WEIGHTS and not n.startswith('training_recipe/')}
    problems = [n for n, d in code.items() if n not in ('submission.py', 'fixed_transfer.py') and d != vendored.get(n)]
    if code.get('submission.py') != expected_sub:
        problems.append('submission.py')
    if code.get('fixed_transfer.py') != fixed:
        problems.append('fixed_transfer.py')
    if problems:
        raise SystemExit(f'final runtime code differs from the vendored runtime transforms: {problems}')
    code_hashes = {n: sha256_bytes(d) for n, d in code.items()}
    ready_equal = None
    if depth == 4:
        ready_equal = code_hashes == READY_CODE_MEMBERS
    states = {}
    for name in WEIGHTS:
        sd = torch.load(io.BytesIO(dict(ma)[name]), map_location='cpu', weights_only=False)['state_dict']
        states[name] = sd_hash(sd)
    audit = {'status': 'COMPLETE', 'candidate': final, 'source_archive': rel(source_zip), 'source_archive_sha256': sha256(source_zip),
             'f5_archive': rel(f5_zip), 'f5_archive_sha256': sha256(f5_zip), 'archive': rel(ma_zip),
             'archive_sha256': sha256(ma_zip), 'runtime': rel(runtime), 'members': [[n, sha256_bytes(d)] for n, d in ma],
             'point_depth': depth, 'deployed_states': states, 'code_members': code_hashes,
             'code_members_equal_ready_zip': ready_equal,
             'transforms': {'F5': 'make_fixed5.py:26-36 (strings read from the pinned file)', suffix: 'make_malloc.py:39-49 (strings read from the pinned file)',
                            'submission_py': ('depth_runtime -> ' if depth != 10 else '') + 'F5 -> ' + suffix},
             'packaging_pins': PACKAGING_PINS, 'lifted_from': LIFTED_FROM_13_15,
             'not_reproduced': 'full81 scoring, C894 qualification, C973 timing, harness replica, READY copy (verification only)'}
    write_json(out / 'final_audit.json', audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('stage')
    p.add_argument('--plan', type=Path, required=True, help='plan driving the packaging (candidate, steps, archives)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trained-plan', type=Path, help='plan the training ran with (resume plan_sha256); default --plan')
    p.add_argument('--training-dir', type=Path, help='default: <plan dir>/training_candidate')
    p.add_argument('--template-zip', type=Path, help="override of plan['teacher_archive']['path'] (sha256 still checked)")
    p.add_argument('--script-spec', type=Path, help='training_script.txt (default: <plan dir>/training_script.txt)')
    p = sub.add_parser('final')
    p.add_argument('--source-zip', type=Path, required=True, help='the stage-15 package ZIP')
    p.add_argument('--candidate', required=True, help='candidate name of the stage-15 package (e.g. SIMSAT2D4FH)')
    p.add_argument('--suffix', choices=['M', 'MA'], default='MA')
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'stage':
        audit = package_stage(args.plan, args.out, args.trained_plan, args.training_dir, args.template_zip, args.script_spec)
    else:
        audit = package_final(args.source_zip, args.out, args.candidate, args.suffix)
    print(json.dumps({'status': 'PACKAGED', 'command': args.command, 'candidate': audit['candidate'], 'members': len(audit['members']),
                      'archive_sha256': audit['archive_sha256']}))


if __name__ == '__main__':
    main()
