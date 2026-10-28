"""Package, evaluate twice, qualify and time the LATCAD full81 fit through the pipeline's own tools.

Steps are ordered so any failure stops before a submission file exists. Nothing is uploaded.
"""
import fcntl,hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'runs/LATCADF_full81';PY=str(R/'.venv310/bin/python')
sys.path.insert(0,str(R/'runs/LATCAD_upperRe'));sys.path.insert(0,str(R/'scripts'))
import package_latcad as P
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def rel(p):return str(Path(p).resolve().relative_to(R))
def locked(*args):
    with (R/'.records/release_event_writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=subprocess.run(['python3','scripts/work_release.py',*args],cwd=R,capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        return p.stdout
def step_package():
    zp=D/'LATCADFH_CANDIDATE_RELEASED_LATTICE_CADENCE_RESEARCH.zip'
    if zp.exists():return zp
    rep=read(D/'training_candidate/report.json');assert rep['status']=='COMPLETE' and rep['steps_actual']==read(D/'candidate_plan.json')['training']['steps']
    recipe={'training_recipe/released_lattice_cadence.json':json.dumps(dict(lattice_phases=[[0,0],[0,1],[1,0],[1,1]],cadence2_probability=0.25,
              cadence_generator='independent CPU generator seed+4',trailing_starts='spacing 1 when fewer than 79 frames remain',
              cadence_counts=rep['cadence_counts'],cadence_trailing_fallbacks=rep['cadence_trailing_fallbacks'],
              parent='C1099H lattice-only full81 plan',endpoint='fixed midpoint/final EMA 50:50'),indent=2).encode(),
            'training_recipe/latcad_train.py.txt':(D/'train.py').read_bytes()}
    states=P.build(D/'training_candidate',D/'candidate_plan.json',zp,recipe)
    import torch;cp=D/'candidate_deployed_checkpoint.pt'
    torch.save(dict(ema_point=states['model.pt'],ema_heads={P.WEIGHTS[n]+'.'+k:v for n,s in states.items() if n!='model.pt' for k,v in s.items()}),cp)
    return zp
def step_eval(zp):
    out={}
    for r in ('r1','r2'):
        name='full81_LATCADFH_'+r;sc=R/'runs/SOUP_20260923'/name/'scores.json'
        if not sc.exists():subprocess.run([PY,str(R/'runs/SOUP_20260923/soup_eval_v2.py'),'--name',name,'--donor',rel(zp),'--cases','runs/LATCAD_full81/full81_cases.json','--input-root','runs/C728R_full81_dense_submission/inputs'],cwd=R,check=True)
        out[r]=read(sc)
    h1={x['case']:x['prediction_hashes'] for x in out['r1']['rows']};h2={x['case']:x['prediction_hashes'] for x in out['r2']['rows']}
    assert h1==h2 and len(h1)==81,'full81 repeat not bit-exact'
    return out['r1'],h1
def step_scaffold(zp,res):
    base=read(R/'runs/C1099R_released_lattice/plan.json');base['candidate']='LATCADFH';base['latcad']='C1099H plan with exact released temporal spacing 2 added (p=0.25); see candidate_plan.json'
    write(D/'plan.json',base)
    with zipfile.ZipFile(zp) as z:members={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    cp=D/'candidate_deployed_checkpoint.pt'
    write(D/'quality_plan.json',dict(archives=dict(candidate=dict(path=rel(zp),sha256=sha(zp),checkpoint=rel(cp),checkpoint_sha256=sha(cp),runtime_members=members))))
    (D/'confirmation').mkdir(exist_ok=True)
    write(D/'confirmation/report.json',dict(candidate='LATCADFH',all_outputs_repeated=True,scores=res['scores'],composite_time90=res['composite_time90'],
          evaluator='runs/SOUP_20260923/soup_eval_v2.py (pipeline-parity verified)',rounds=['runs/SOUP_20260923/full81_LATCADFH_r1/scores.json','runs/SOUP_20260923/full81_LATCADFH_r2/scores.json']))
    write(D/'complete.json',dict(status='COMPLETE',composite_time90=res['composite_time90']))
def step_register():
    spec=D/'spec_LATCADFH.json'
    if not spec.exists():
        write(spec,dict(candidate='LATCADFH',objective='Full81 fit of lattice plus released temporal spacing 2, for a submission candidate.',comparison='Versus C1099H (lattice-only full81) and C974H/C1018H; same runtime bytes, same packaging.',
              scale='18464 updates, all 81 clean cases.',finish='Full81 quality twice, deployment qualification, C974 timing, then a clearly named candidate file; no upload.',
              budget='Existing shared A5500; no paid resources.',next_action='Qualify and time the packaged archive.'))
    print(locked('start','--secondary','--spec',rel(spec)))
if __name__=='__main__':
    zp=step_package();print('packaged',rel(zp),sha(zp))
    res,h=step_eval(zp);print('full81 composite_time90',res['composite_time90'],'| repeat bit-exact over 81 cases')
    step_scaffold(zp,res);step_register();print('scaffold + LATCADH registered; next: qualification')
