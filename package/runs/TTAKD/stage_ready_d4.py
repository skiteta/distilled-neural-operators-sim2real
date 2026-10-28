"""Stage a 4-block (D4) host-allocation (MA) candidate as READY and write its pre-upload forecast.
Usage: stage_ready_d4.py CAND_MA SOURCE_F5 SOURCE_WEIGHTS HELD_JSON REPLICA_JSON REF10_MA
  e.g. SIMSAT2D4HF5MA SIMSAT2D4HF5 SIMSAT2D4H runs/TTAKD/result_SIMSAT2D4_vsSIMA.json runs/HARNESS_20260925/ready_zip_ab_SIMSAT2D4_MA.json SIMSATHF5MA
Checks on the actual files (as stage_ready_m.py): deployment proof on the source zip and the copied READY file; qualification
PASS; accepted timing; CAND differs from SOURCE_F5 only in submission.py, by exactly the make_malloc.py MA block; full81
prediction hashes equal SOURCE_F5. Depth-specific: SOURCE_F5 differs from the Public-run SIMAHF5 runtime only in weights and in
submission.py, where submission.py == depth_runtime(SIMAHF5 submission.py) (the one-line block-count assert); the packaged
point model has depth 4.
Forecast = the 10-block MA forecast (Public-measured LC4BH quality + SIMA share + held gain vs SIMA x 0.40, residual saving s)
with the single-pass predict reduced by d = (1 - r_cand / r_ref10) x 1.426 ms/window, r = geometric warm ratio of the C973
timing against the same fixed control (REF10_MA = the 10-block MA candidate measured the same way)."""
import sys, json, shutil, hashlib, datetime, math, zipfile, io
from pathlib import Path
sys.path.insert(0, 'scripts')
from submission_release_gate import check_proof
from experiment_records import register
from depth_distillation_cycle_v2 import depth_runtime
import torch
R = Path(__file__).resolve().parents[2]; cand, source, weights, held_json, replica_json, ref10 = sys.argv[1:7]
assert cand.endswith('MA')
S_C, S_LO, S_HI = 1.5, 0.5, 3.0   # MA residual saving per window (ms): central / low / high, as for the 10-block MA candidates
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
L = R / 'runs/EXACTLANE' / cand; f = json.load(open(L / 'full81_result.json')); src = f['zip']
proof = f'runs/{cand}Q_deployment_validation/deployment_gate_proof.json'; check_proof(R, src, proof)
m = json.load(open(R / f'runs/{cand}T_public_runtime/measurement.json')); assert m['accepted_timing']
q = json.load(open(R / f'runs/{cand}Q_deployment_validation/qualification_complete.json')); assert q['status'] == 'PASS' and q['archive_sha256'] == f['sha256']
assert f['prediction_hashes_equal_source'] is True and f['source'] == source
BLOCK = Path(R / 'runs/EXACTLANE/make_malloc.py').read_text().split("BLOCK = '''", 1)[1].split("'''", 1)[0]
top = "    _LIBC.mallopt(-2, 64 * 1024 * 1024)    # M_TOP_PAD\n"
BLOCK = BLOCK.replace(top, top + "    _LIBC.mallopt(-8, 1)                   # M_ARENA_MAX: helper threads (e.g. a queue feeder) share the prepared heap\n")
fs = json.load(open(R / f'runs/EXACTLANE/{source}/full81_result.json'))
with zipfile.ZipFile(R / src) as a, zipfile.ZipFile(R / fs['zip']) as b:
    assert a.namelist() == b.namelist()
    diff = [n for n in a.namelist() if a.read(n) != b.read(n)]; assert diff == ['submission.py'], diff
    s_new, s_old = a.read('submission.py').decode(), b.read('submission.py').decode()
    assert s_new == s_old.replace('from __future__ import annotations\n', 'from __future__ import annotations\n' + BLOCK, 1)
ref = zipfile.ZipFile(R / 'runs/EXACTLANE/SIMAHF5/SIMAHF5_FIXED5_RESEARCH.zip'); old = zipfile.ZipFile(R / fs['zip'])
hz = lambda z: {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.filename.startswith('training_recipe/')}
ha, hb = hz(ref), hz(old); assert set(ha) == set(hb)
changed = {k for k in ha if ha[k] != hb[k]}
assert changed <= {'center_head.pt', 'hidden_head.pt', 'model.pt', 'observed_center_head.pt', 'radius_head.pt', 'submission.py'}, changed
payload = torch.load(io.BytesIO(old.read('model.pt')), map_location='cpu', weights_only=False)
NOMS = bool(payload['model_config'].get('no_multiscale'))   # SIMSAT2D4N: runtime also drops the multiscale context
expected_sub = depth_runtime(ref.read('submission.py'))
if NOMS:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('f10', R / 'runs/SD10/finalize_sd10.py'); _F10 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F10)
    expected_sub = _F10.noms_runtime(expected_sub)
    assert not any(k.startswith('multiscale.') for k in payload['state_dict']), 'no-multiscale model still has multiscale weights'
assert old.read('submission.py') == expected_sub, 'submission.py is not the expected transform of SIMAHF5'
depth = int(payload['model_config']['depth']); assert depth in (3, 4), depth   # SIMSAT2D3: 3 blocks
n = {'upperRe': 29, 'AoA15': 17, 'AoA20': 16}; hj = json.load(open(R / held_json))
g = sum(n[t] * hj['surfaces'][t]['vs_SIMA'] for t in n) / 62
ratios = [r['geometric_warm_ratio'] for r in m['rounds']]
m10 = json.load(open(R / f'runs/{ref10}T_public_runtime/measurement.json')); ratios10 = [r['geometric_warm_ratio'] for r in m10['rounds']]
c0 = m['pairs'][0]['records']['control']['archive_sha256']; c1 = m10['pairs'][0]['records']['control']['archive_sha256']; assert c0 == c1, 'different timing controls'
rc, r10 = sum(ratios) / len(ratios), sum(ratios10) / len(ratios10)
d = (1 - rc / r10) * 1.426; d_lo, d_hi = 0.75 * d, 1.25 * d
replica = json.load(open(R / replica_json))['summary']
ev = dict(candidate=cand, kind='performance', valid=True, guardrails_pass=True, visual=False, is_best=False,
  summary=(f"{cand} = {source} (4 blocks, single-pass inference at fixed batch 5) + host memory allocation policy (predictions unchanged). full81: prediction hashes identical to {source} ({f['full81_parts5']:.6f}). "
           f"Timing (C973, same control): resident ratio {rc:.4f} (10-block MA {ref10} {r10:.4f}) -> predict about {d:.2f} ms/window faster (server-equivalent). "
           f"Harness replica ({replica_json}), total time of one 5-window call: Pipe {replica['A/pipe/raw']['total_ms_median']:.2f}->{replica['B/pipe/raw']['total_ms_median']:.2f} ms, Queue {replica['A/queue/raw']['total_ms_median']:.2f}->{replica['B/queue/raw']['total_ms_median']:.2f} ms."),
  metric=dict(baseline=f['source_full81'], baseline_id=source, candidate=f['full81_parts5'], direction='max', evaluation_id='C983_C728_full81_training_exposed_v1_inferred_time90_v1', evaluation_ref='runs/C1017R_backbone_dropout_composite/derived_evaluation.json', name='inferred_composite_fixed_time90'),
  issue='The official harness is not public, so the MA saving on the server is unconfirmed. The 4-block quality difference is the held 3-surface verdict (0.40x on the Public scale).',
  next_action='Fix the pre-upload forecast.',
  critical_check=dict(risk='Failure or prediction change because the 4-block runtime changes (block-count assert).', finding='submission.py equals SIMAHF5 with depth_runtime applied, point model depth 4, standard checks PASS, predictions bit-identical to the F5 source candidate.', evidence=proof),
  evidence=[f'runs/EXACTLANE/{cand}/full81_result.json', proof, f'runs/{cand}Q_deployment_validation/qualification_complete.json', f'runs/{cand}T_public_runtime/measurement.json', replica_json, held_json])
print(register(ev, f'runs/EXACTLANE/{cand}/performance_event.json'))
jst = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)).strftime('%Y%m%d_%H%M%S'); name = f'{jst}_JST__READY__{cand}_FIXED5.zip'; assert len(name) <= 64
dst = R / 'submissions' / name; shutil.copyfile(R / src, dst); assert h(dst) == f['sha256']; check_proof(R, 'submissions/' + name, proof)
ev = dict(candidate=cand, kind='delivery', candidate_status='ready', package_stage='local_validated', announce=False, visual=False, is_best=False, artifacts=['submissions/' + name], deployment_gate=proof,
  summary=f"{cand} made submittable: {weights} weights (4 blocks) + single-pass inference at fixed batch 5 + host memory allocation policy. Standard checks PASS, predictions identical to {source}, timing resident ratios {ratios[0]:.3f}/{ratios[1]:.3f}.",
  issue='Time saving on Public unconfirmed.', next_action='Fix the pre-upload forecast.',
  critical_check=dict(risk='Failure from the runtime change.', finding='C894 standard checks (pip and official image, twice each, strict210, official scoring, split invariance) PASS, float32 output, predictions bit-identical to the source candidate.', evidence=proof),
  evidence=[f'runs/EXACTLANE/{cand}/performance_event.json', proof, f'runs/{cand}Q_deployment_validation/qualification_complete.json', f'runs/{cand}T_public_runtime/measurement.json'],
  scope=dict(execution='Full81 (parts of 5 twice, whole), C894 protocol, 12 timing pairs, local harness replica.', review='Same AI.'))
print(name, register(ev, f'runs/EXACTLANE/{cand}/delivery_event.json'))
w = (.5, .1, .1, .1, .2); p = .6854; comp = lambda s: sum(a * b ** p for a, b in zip(w, s)) ** (1 / p)
qv = [94.684980, 79.179663, 94.211115, None, 44.115534]; lc = lambda T: comp([qv[0], qv[1], qv[2], T, qv[4]])
T = lambda t: 100 / (1 + math.sqrt(t / 728.96)); tF5 = (6.204, 6.346, 6.488)   # 10-block single-pass F5 total per window [low, central, high]
central = lc(T(tF5[1] - d - S_C)) + 0.0127 + 0.40 * g
low = lc(T(tF5[2] - d_lo - S_LO)) + 0.0127 - 0.012 + 0.2 * g
high = lc(T(tF5[0] - d_hi - S_HI)) + 0.0127 + 0.012 + 0.7 * g
sd = math.sqrt(((high - low) / 3.29) ** 2 + 0.012 ** 2)
P = lambda x: 0.5 * (1 + math.erf((central - x) / (sd * math.sqrt(2))))
pre = dict(kind='public_upload_preregistration', fixed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), slot='2026-09-26 UTC or 2026-09-27 UTC', candidate=cand, artifact='submissions/' + name, sha256=h(dst),
  method=(f'Quality: Public-measured LC4BH + SIMA share +0.0127 + held gain vs SIMA {g:+.4f} x 0.40 (0.2-0.7). Time: 10-block single-pass F5 total 6.346 ms/window minus the 4-block predict saving '
          f'{d:.3f} ms/window (C973 ratio {rc:.4f} vs {ref10} {r10:.4f} against the same control, x 1.426 ms/window server F5 predict; range x0.75-1.25) minus the host-allocation saving s (central {S_C}; range {S_LO}-{S_HI}).'),
  forecast=dict(final_central=round(central, 3), final_range=[round(low, 3), round(high, 3)], time_central=round(T(tF5[1] - d - S_C), 3), time_range=[round(T(tF5[2] - d_lo - S_LO), 3), round(T(tF5[0] - d_hi - S_HI), 3)],
    P_at_or_above_top10_line_81_775=f'about {round(100 * P(81.77487))} percent (line as of 2026-09-25 16:05 UTC; it can rise)',
    P_at_or_above_top5_81_968=f'about {round(100 * P(81.96799))} percent', P_82=f'about {round(100 * P(82.0))} percent'),
  what_it_decides={'time_score': 'MA saving and 4-block predict saving on the server', 'quality subscores': 'held 4-block student quality transferred at 0.40'},
  leaderboard_rule='Force_Best', not_claimed=['82', 'top-10 certainty'], upload='manual')
Path(R / f'runs/NEXTSLOT_20260924/preregistration_{cand}.json').write_text(json.dumps(pre, indent=1, ensure_ascii=False) + '\n'); print(json.dumps(pre['forecast'], indent=1, ensure_ascii=False))
