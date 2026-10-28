"""Package SD10 fits (runtime bytes unchanged, weights replaced), score held/full81 against teachers and the other
candidates with the parity-verified evaluator, and apply the pre-fixed rule in runs/SD10/decision_rule.json."""
import hashlib,io,json,subprocess,sys,zipfile,numpy as np,torch
from pathlib import Path
R=Path(__file__).resolve().parents[2];PY=str(R/'.venv310/bin/python');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from realpde.metrics import inferred_composite_score as comp
WEIGHTS={'model.pt':'point','center_head.pt':'center','radius_head.pt':'radius','hidden_head.pt':'hidden','observed_center_head.pt':'observed'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).resolve().relative_to(R))
NOMS_OLD='    model = HybridFlowOperator(ModelConfig.from_dict(checkpoint["model_config"])).to(_DEVICE)\n'
NOMS_NEW=NOMS_OLD+'    if checkpoint["model_config"].get("no_multiscale"):\n        model.multiscale = torch.nn.Identity()   # student trained without the multiscale context\n'
def noms_runtime(data):
    text=data.decode();assert text.count(NOMS_OLD)==1
    return text.replace(NOMS_OLD,NOMS_NEW).encode()


def package(D,teacher_path_override=None,final_weight=0.5):
    tag='' if final_weight==0.5 else f'_EMA{int(round(final_weight*100)):03d}'   # EMA-mix variants get their own files; 0.5 = historical default, unchanged
    m=read(D/'plan.json');t=D/'training_candidate';rep=read(t/'report.json');target=D/(m['candidate']+tag+'_SD10_RESEARCH.zip');cp=D/(m['candidate']+tag+'_deployed_checkpoint.pt')
    if target.exists():return target
    assert rep['status']=='COMPLETE' and rep['teacher_unchanged'] and (rep['fine_source_and_reflection_stream_exact_to_teacher_fit'] or not m['training'].get('replay_teacher_stream',True)) and rep['student_depth']==int(m['training'].get('student_depth',10))
    end=torch.load(t/f"resume_{m['training']['steps']}.pt",map_location='cpu',weights_only=False);mid=torch.load(t/f"resume_{m['middle_step']}.pt",map_location='cpu',weights_only=False)
    assert end['plan_sha256']==mid['plan_sha256']==sha(D/'plan.json') and end['model_config']['depth']==int(m['training'].get('student_depth',10))
    wf=float(final_weight);av=(lambda a,b:{k:(a[k]*.5+b[k]*.5 if a[k].is_floating_point() or a[k].is_complex() else a[k].clone()) for k in a}) if wf==0.5 else (lambda a,b:{k:(a[k]*(1-wf)+b[k]*wf if a[k].is_floating_point() or a[k].is_complex() else a[k].clone()) for k in a})
    point=av(mid['ema_point'],end['ema_point']);heads=av(mid['ema_heads'],end['ema_heads'])
    torch.save(dict(ema_point=point,ema_heads=heads,model_config=end['model_config'],step=end['step']),cp)
    teacher=m['teacher_archive'];fast=m['runtime_baseline'];tpath=R/(teacher_path_override or teacher['path'])
    assert sha(tpath)==teacher['sha256'] and sha(R/fast['path'])==fast['sha256'],'teacher archive hash differs from the plan record'
    with zipfile.ZipFile(tpath) as old,zipfile.ZipFile(R/fast['path']) as runtime,zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in runtime.namelist():
            if name.startswith('training_recipe/'):continue
            data=runtime.read(name)
            SDEP=int(m['training'].get('student_depth',10))
            if name=='submission.py' and SDEP!=10:
                from depth_distillation_cycle_v2 import depth_runtime
                data=depth_runtime(data)
            if name=='submission.py' and m['training'].get('no_multiscale'):
                data=noms_runtime(data)   # SIMSAT2D4N: drop the multiscale context before loading (flag in model_config)
            if name in WEIGHTS:
                payload=torch.load(io.BytesIO(old.read(name)),map_location='cpu',weights_only=False);key=WEIGHTS[name]
                payload['state_dict']=point if key=='point' else {k[len(key)+1:]:v for k,v in heads.items() if k.startswith(key+'.')}
                if key=='point':payload['model_config']=end['model_config']
                b=io.BytesIO();torch.save(payload,b);data=b.getvalue()
            z.writestr(name,data)
        written={n for n in runtime.namelist() if not n.startswith('training_recipe/')}
        for name in old.namelist():
            if name.startswith('training_recipe/') and name not in written:z.writestr(name,old.read(name));written.add(name)
        cand=m['candidate']
        # the script that actually trained this run (runs/<run>/training_script.txt: first line = script, further lines = helper modules)
        spec=[x.strip() for x in (D/'training_script.txt').read_text().splitlines() if x.strip()] if (D/'training_script.txt').exists() else ['runs/SD10/train_sd10.py']
        recipe=[(f'training_recipe/{cand}_continuation.json',json.dumps(m,ensure_ascii=False,indent=2).encode()),(f'training_recipe/{cand}_train.py.txt',(R/spec[0]).read_bytes())]
        recipe+=[(f'training_recipe/{cand}_{Path(x).name}.txt',(R/x).read_bytes()) for x in spec[1:]]
        for name,data in recipe:
            assert name not in written,name;z.writestr(name,data);written.add(name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())),'duplicate ZIP member names'
        with zipfile.ZipFile(R/fast['path']) as rt:
            for n in rt.namelist():
                if n not in WEIGHTS and not n.startswith('training_recipe/') and not (n=='submission.py' and int(m['training'].get('student_depth',10))!=10):assert z.read(n)==rt.read(n),n   # executable runtime bytes unchanged
        for name,key in WEIGHTS.items():
            sd=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)['state_dict'];exp=point if key=='point' else {k[len(key)+1:]:v for k,v in heads.items() if k.startswith(key+'.')}
            assert sd.keys()==exp.keys() and all(torch.equal(v,exp[k]) for k,v in sd.items())
    return target
def score(name,donor,cases):
    sc=R/'runs/SOUP_20260923'/name/'scores.json'
    if not sc.exists():subprocess.run([PY,str(R/'runs/SOUP_20260923/soup_eval_v2.py'),'--name',name,'--donor',donor,'--cases',cases,'--input-root','runs/C728R_full81_dense_submission/inputs'],cwd=R,check=True,capture_output=True)
    return read(sc)
def agg(rows):
    n=sum(r['windows'] for r in rows);cnt=sum(r['scored_elements'] for r in rows)
    s={k+'_score':100/(1+.5*sum(r['mean_errors'][k]*r['windows'] for r in rows)/n) for k in ('rel_l2','tke','mvpe')}
    s['sps_score']=sum(r['sps_score']*r['scored_elements'] for r in rows)/cnt
    return float(comp(s['rel_l2_score'],s['tke_score'],s['mvpe_score'],90.0,s['sps_score']))
def paired(A,B):
    A={r['case']:r for r in A['rows']};B={r['case']:r for r in B['rows']};cs=sorted(A);assert cs==sorted(B)
    d=agg([B[c] for c in cs])-agg([A[c] for c in cs]);rng=np.random.default_rng(20260923)
    reps=[agg([B[c] for c in p])-agg([A[c] for c in p]) for p in ([cs[i] for i in rng.integers(0,len(cs),len(cs))] for _ in range(2000))]
    return d,[float(x) for x in np.quantile(reps,[.05,.95])]
SURF={'upperRe':(29,'C1096R','C1014R_backbone_dropout_composite/C1014B_DROPOUT_COMPOSITE_RESEARCH.zip','C1014B','C1096R_released_lattice/C1096B_CANDIDATE_RELEASED_LATTICE_RESEARCH.zip','C1096B','C1048R_fused_half_return/C1048B_HALF_CONV_RESEARCH.zip','C1048B',82.430590),
      'AoA15':(17,'C1097R','C1015R_backbone_dropout_composite/C1015B_DROPOUT_COMPOSITE_RESEARCH.zip','C1015B','C1097R_released_lattice/C1097B_CANDIDATE_RELEASED_LATTICE_RESEARCH.zip','C1097B','C1049R_fused_half_return/C1049B_HALF_CONV_RESEARCH.zip','C1049B',82.876976),
      'AoA20':(16,'C1098R','C1016R_backbone_dropout_composite/C1016B_DROPOUT_COMPOSITE_RESEARCH.zip','C1016B','C1098R_released_lattice/C1098B_CANDIDATE_RELEASED_LATTICE_RESEARCH.zip','C1098B','C1050R_fused_half_return/C1050B_HALF_CONV_RESEARCH.zip','C1050B',81.574948)}
def held(lin):
    out={}
    for tag,(n,tr,dz,did,lz,lid,cz,cid,base) in SURF.items():
        cases=f'runs/{tr}_released_lattice/eval_groups.json';D=R/f'runs/SD10_{lin}_{tag}'
        if not (D/'training_candidate/report.json').exists():out[tag]=None;continue
        zp=package(D);S=score(f'{tag}_SD10{lin}B',rel(zp),cases)
        tz,tid=(dz,did) if lin=='D' else (lz,lid);T=score(f'{tag}_parity_{tid}',f'runs/{tz}',cases)
        d,ci=paired(T,S);out[tag]=dict(n=n,SD10=S['composite_time90'],teacher=tid,teacher_score=T['composite_time90'],vs_teacher=d,ci90=ci,vs_C974_lineage=S['composite_time90']-base)
    return out
def full81(lin):
    D=R/f'runs/SD10_{lin}_full81'
    if not (D/'training_candidate/report.json').exists():return None
    zp=package(D);cases='runs/LATCAD_full81/full81_cases.json'
    r1=score(f'full81_SD10{lin}H_r1',rel(zp),cases);r2=score(f'full81_SD10{lin}H_r2',rel(zp),cases)
    assert {x['case']:x['prediction_hashes'] for x in r1['rows']}=={x['case']:x['prediction_hashes'] for x in r2['rows']},'repeat not bit-exact'
    tz='runs/C1017R_backbone_dropout_composite/C1017H_DROPOUT_COMPOSITE_RESEARCH.zip' if lin=='D' else 'runs/C1099R_released_lattice/C1099H_CANDIDATE_RELEASED_LATTICE_RESEARCH.zip'
    T=score('full81_C1017H_r1' if lin=='D' else 'full81_C1099H_r1',tz,cases);d,ci=paired(T,r1)
    return dict(zip=rel(zp),sha256=sha(zp),full81=r1['composite_time90'],teacher=T['composite_time90'],vs_teacher=d,ci90=ci,vs_C974=r1['composite_time90']-85.757160)
if __name__=='__main__':
    res={}
    for lin in sys.argv[1:]:
        res[lin]=dict(held=held(lin),full81=full81(lin))
        (R/f'runs/SD10/result_{lin}.json').write_text(json.dumps(res[lin],indent=1,default=float))
        print(lin,json.dumps(res[lin],default=lambda x:round(float(x),6))[:1500])
