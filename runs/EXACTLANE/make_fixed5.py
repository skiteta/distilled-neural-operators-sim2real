"""Fixed-batch-5 variant of a qualified candidate: network batch 3 -> 5 (official parts are 5 windows: one replay, no padding),
fixed_transfer.py (pinned host set allocated in the warm-up, GPU-side checks), stderr log line. float32 outputs, split-invariant.
Evaluates full81 in official parts of 5 twice (bit-exact repeat) and whole cases once, then runs the standard C894
qualification and C973 timing. Usage: make_exactlane.py SOURCE_RUN_DIR [SUFFIX]"""
import json,hashlib,zipfile,subprocess,sys,difflib,shutil,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[2];PY=str(R/'.venv310/bin/python')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
src_run=R/sys.argv[1];q=read(src_run/'qsource/quality_plan.json');ar=q['archives']['candidate'];src_zip=R/ar['path'];assert sha(src_zip)==ar['sha256']
plan=read(src_run/'qsource/plan.json');base=plan['candidate'];cand=base+(sys.argv[2] if len(sys.argv)>2 else 'F5')
D=R/f'runs/EXACTLANE/{cand}';D.mkdir(parents=True);zp=D/f'{cand}_FIXED5_RESEARCH.zip'
old="def predict(input_array: np.ndarray, metadata=None):\n    return _predict(input_array)\n"
new=("def predict(input_array: np.ndarray, metadata=None):\n"
     "    import sys as _sys, time as _time\n"
     "    _t0 = _time.perf_counter()\n"
     "    result = _predict(input_array)\n"
     "    print(f\"realpde_batch shape={tuple(np.shape(input_array))} dtype={result['prediction'].dtype} predict_ms={1000*(_time.perf_counter()-_t0):.2f}\", file=_sys.stderr, flush=True)\n"
     "    return result\n")
tail=("\n# Fixed network batch 5 (official scored parts are 5 windows): pinned host sets allocated in the warm-up, checks on the GPU.\n"
      "from fixed_transfer import install as _install_fixed_transfer\n_install_fixed_transfer(_TransferNamespace(globals()))\n")
with zipfile.ZipFile(src_zip) as a,zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as b:
    for info in a.infolist():
        data=a.read(info.filename)
        if info.filename=='submission.py':
            s=data.decode();assert s.count(old)==1 and s.count('_install_async_transfer(_TransferNamespace(globals()))')==1
            s2=s.replace(old,new)
            for a_,b_ in (("_INTERNAL_BATCH_SIZE = 3\n","_INTERNAL_BATCH_SIZE = 5\n"),("_TRANSFER_GROUP_SIZE = 48  # bounded GPU staging; a multiple of fixed network batch3","_TRANSFER_GROUP_SIZE = 50  # bounded GPU staging; a multiple of fixed network batch5")):
                assert s2.count(a_)==1,a_;s2=s2.replace(a_,b_)
            s2=s2+tail;data=s2.encode()
            added=[l for l in difflib.unified_diff(s.splitlines(),s2.splitlines(),lineterm='',n=0) if l[:1]=='+' and not l.startswith('+++')]
            removed=[l for l in difflib.unified_diff(s.splitlines(),s2.splitlines(),lineterm='',n=0) if l[:1]=='-' and not l.startswith('---')]
            assert sorted(removed)==sorted(['-    return _predict(input_array)','-_INTERNAL_BATCH_SIZE = 3','-_TRANSFER_GROUP_SIZE = 48  # bounded GPU staging; a multiple of fixed network batch3']),removed
        b.writestr(info.filename,data)
    b.writestr('fixed_transfer.py',(R/'runs/EXACTLANE/fixed_transfer.py').read_bytes())
with zipfile.ZipFile(zp) as z,zipfile.ZipFile(src_zip) as a:
    n=z.namelist();assert len(n)==len(set(n)) and z.testzip() is None
    members={x:hashlib.sha256(z.read(x)).hexdigest() for x in n}
    assert all(members[x]==hashlib.sha256(a.read(x)).hexdigest() for x in n if x not in ('submission.py','fixed_transfer.py'))
print('built',zp.relative_to(R),sha(zp)[:16],flush=True)
rel=str(zp.relative_to(R));cases='runs/LATCADF_full81/full81_cases.json';res={}
for tag,ev in (('p5r1','soup_eval_v4_parts5.py'),('p5r2','soup_eval_v4_parts5.py'),('whole','soup_eval_v2.py')):
    name=f'full81_{cand}_{tag}'
    if not (R/'runs/SOUP_20260923'/name/'scores.json').exists():
        subprocess.run([PY,str(R/'runs/SOUP_20260923'/ev),'--name',name,'--donor',rel,'--cases',cases,'--input-root','runs/C728R_full81_dense_submission/inputs'],cwd=R,check=True,capture_output=True)
    res[tag]=read(R/'runs/SOUP_20260923'/name/'scores.json')
assert {x['case']:x['prediction_hashes'] for x in res['p5r1']['rows']}=={x['case']:x['prediction_hashes'] for x in res['p5r2']['rows']},'parts-of-5 repeat not bit-exact'
assert {x['case']:x['prediction_hashes'] for x in res['p5r1']['rows']}=={x['case']:x['prediction_hashes'] for x in res['whole']['rows']},'parts-of-5 and whole-case calls differ (split invariance broken)'
srcs=read(R/f'runs/SOUP_20260923/full81_{base}_r1/scores.json')
out=dict(candidate=cand,source=base,zip=rel,sha256=sha(zp),full81_parts5=res['p5r1']['composite_time90'],full81_whole=res['whole']['composite_time90'],source_full81=srcs['composite_time90'],
         delta_parts5=res['p5r1']['composite_time90']-srcs['composite_time90'],delta_whole=res['whole']['composite_time90']-srcs['composite_time90'],
         subscores_parts5=res['p5r1']['scores'],source_subscores=srcs['scores'])
write(D/'full81_result.json',out);print('full81',json.dumps({k:out[k] for k in ('full81_parts5','full81_whole','source_full81','delta_parts5','delta_whole')}),flush=True)
src=D/'qsource';shutil.copytree(src_run/'qsource',src)
plan['candidate']=cand;plan['fixed5']=dict(source_candidate=base,source_archive=ar['path'],change='network batch 3 -> 5 (split-invariant), warm-up-allocated pinned host set, GPU-side checks; stderr log line');write(src/'plan.json',plan)
ar2=dict(ar);ar2.update(path=rel,sha256=sha(zp),runtime_members=members);write(src/'quality_plan.json',dict(archives=dict(candidate=ar2)))
rep=read(src/'confirmation/report.json');rep.update(candidate=cand,scores=res['whole']['scores'],composite_time90=res['whole']['composite_time90'],rounds=[f'runs/SOUP_20260923/full81_{cand}_whole/scores.json'])
write(src/'confirmation/report.json',rep);write(src/'complete.json',dict(status='COMPLETE',composite_time90=res['whole']['composite_time90']))
spec=D/f'spec_{cand}.json';write(spec,dict(candidate=cand,objective=f'{base} with a fixed network batch of 5 matching the official 5-window parts (one replay, no padding), float32 outputs, split-invariant.',comparison=f'Full81 (parts of 5 and whole cases) vs {base}; standard qualification; timing.',scale='Inference only.',finish='C894 qualification, C973 timing, READY file.',budget='Existing A5500.',next_action='Qualify and time.'))
with (R/'.records/release_event_writer.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);subprocess.run(['python3','scripts/work_release.py','start','--secondary','--spec',str(spec.relative_to(R))],cwd=R,capture_output=True)
qd=f'runs/{cand}Q_deployment_validation';td=f'runs/{cand}T_public_runtime'
qq=subprocess.run([PY,'scripts/qualify_learning_candidate.py','--source-run',str(src.relative_to(R)),'--run',qd,'--prior-cost','runs/SD4DQ_deployment_validation/budget_live.json'],cwd=R)
print('qualification exit',qq.returncode,flush=True)
if qq.returncode==0:
    t=subprocess.run([PY,'runs/C1099R_released_lattice/prepare_timing.py','--source-run',str(src.relative_to(R)),'--qualification',qd,'--out',td],cwd=R)
    print('timing exit',t.returncode,flush=True)
