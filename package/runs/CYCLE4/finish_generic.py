"""Generic full81 finisher: package (any depth; runtime patched only for depth != 10), full81 twice bit-exact, paired vs a
teacher full81 score, scaffold qsource, deployment qualification, C973 timing. Usage: finish_generic.py RUN_DIR CAND TEACHER_SCORE_NAME"""
import json,subprocess,sys,hashlib,zipfile,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];PY=str(R/'.venv310/bin/python')
spec=importlib.util.spec_from_file_location('f10',R/'runs/SD10/finalize_sd10.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def w(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=float)+'\n')
D=R/sys.argv[1];cand=sys.argv[2];tname=sys.argv[3]
zp=F.package(D);rel=str(zp.relative_to(R));cases='runs/LATCADF_full81/full81_cases.json'
n=zipfile.ZipFile(zp).namelist();assert len(n)==len(set(n)),'duplicate members'
r1=F.score(f'full81_{cand}_r1',rel,cases);r2=F.score(f'full81_{cand}_r2',rel,cases)
assert {x['case']:x['prediction_hashes'] for x in r1['rows']}=={x['case']:x['prediction_hashes'] for x in r2['rows']},'repeat not bit-exact'
T=json.loads((R/'runs/SOUP_20260923'/tname/'scores.json').read_text());d,ci=F.paired(T,r1)
res=dict(candidate=cand,zip=rel,sha256=sha(zp),full81=r1['composite_time90'],vs_teacher=d,ci90=ci,vs_C974=r1['composite_time90']-85.757160,teacher_scores=tname)
w(D/'full81_result.json',res);print('full81',json.dumps(res),flush=True)
m=json.loads((D/'plan.json').read_text());cp=D/f'{cand}_deployed_checkpoint.pt'
base=json.loads((R/'runs/C1099R_released_lattice/plan.json').read_text());base['candidate']=cand;base['source_plan']=str((D/'plan.json').relative_to(R))
src=D/'qsource';(src/'confirmation').mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zp) as z:members={x:hashlib.sha256(z.read(x)).hexdigest() for x in z.namelist()}
w(src/'plan.json',base);w(src/'quality_plan.json',dict(archives=dict(candidate=dict(path=rel,sha256=sha(zp),checkpoint=str(cp.relative_to(R)),checkpoint_sha256=sha(cp),runtime_members=members))))
w(src/'confirmation/report.json',dict(candidate=cand,all_outputs_repeated=True,scores=r1['scores'],composite_time90=r1['composite_time90'],evaluator='runs/SOUP_20260923/soup_eval_v2.py',rounds=[f'runs/SOUP_20260923/full81_{cand}_r1/scores.json',f'runs/SOUP_20260923/full81_{cand}_r2/scores.json']))
w(src/'complete.json',dict(status='COMPLETE',composite_time90=r1['composite_time90']))
qd=f'runs/{cand}Q_deployment_validation';td=f'runs/{cand}T_public_runtime'
q=subprocess.run([PY,'scripts/qualify_learning_candidate.py','--source-run',str(src.relative_to(R)),'--run',qd,'--prior-cost','runs/SD4DQ_deployment_validation/budget_live.json'],cwd=R)
print('qualification exit',q.returncode,flush=True)
if q.returncode==0:
    t=subprocess.run([PY,'runs/C1099R_released_lattice/prepare_timing.py','--source-run',str(src.relative_to(R)),'--qualification',qd,'--out',td],cwd=R)
    print('timing exit',t.returncode,flush=True)
