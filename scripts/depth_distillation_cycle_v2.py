"""Full same-fold depth compression with direct Public-runtime qualification."""
from pathlib import Path
import copy,datetime,hashlib,importlib.util,io,json,os,shutil,subprocess,sys,time,zipfile
from experiment_records import read,write,sha,rel,register
from point_interval_factorial_v2 import mutation
from submission_release_gate import check_proof
import complete_recovery_and_rotation_cycles as resources

R=Path(__file__).resolve().parents[1];D=R/'runs/C1028R_depth_distillation'
PREVIOUS=R/'runs/C1019R_simulation_replay'
PY=str(R/'.venv310/bin/python');ENV=dict(resources.ENV,PYTHONDONTWRITEBYTECODE='1')
CONFIG=[('C1028B','C1014R_backbone_dropout_composite'),('C1029B','C1015R_backbone_dropout_composite'),('C1030B','C1016R_backbone_dropout_composite'),('C1031H','C1017R_backbone_dropout_composite')]
WEIGHTS={'model.pt':'point','center_head.pt':'center','radius_head.pt':'radius','hidden_head.pt':'hidden','observed_center_head.pt':'observed'}
def folder(cid):return R/'runs'/(cid[:-1]+'R_depth_distillation')


def depth_runtime(data):
    text=data.decode()
    old='assert len(_LOCAL_BLOCKS) == 12'
    new='assert len(_LOCAL_BLOCKS) == int(_MODEL.config.depth) + sum(isinstance(v, OperatorBlock) for v in _MODEL.multiscale.modules())'
    assert text.count(old)==1
    return text.replace(old,new).encode()



def prepare():
    assert not D.exists()
    fast=read(R/'runs/C974R_async_startup_confirmation/stock_decision.json')['actual_same_zip']
    for cid,parent_name in CONFIG:
        d=folder(cid);d.mkdir();parent=R/'runs'/parent_name;old=read(parent/'plan.json')
        q=read(parent/'quality_plan.json');teacher=q['archives']['candidate']
        assert sha(R/teacher['path'])==teacher['sha256']
        for name in ['config.json','inputs_index.json','train_groups.json','eval_groups.json','tuning_groups.json','quality.py','reflection_transform.py']:
            shutil.copy2(parent/name,d/name)
        rt=d/'teacher_runtime';rt.mkdir()
        with zipfile.ZipFile(R/teacher['path']) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(rt)
        fine=old['training']['steps'];warm=fine//2
        m={k:copy.deepcopy(old[k]) for k in ['evaluation_id','evaluation_ref','boundary','train_cases','eval_cases','full81','sampling','derived_evaluation_id','derived_evaluation_ref']}
        m.update(candidate=cid,baseline=old['candidate'],teacher_source=rel(parent),teacher_archive=teacher,runtime=rel(rt),
                 reflection_source=rel(d/'reflection_transform.py'),runtime_baseline=fast,
                 historical_control_rows=rel(parent/'confirmation/quality/candidate'),
                 objective='Preserve admitted same-fold teacher quality while reducing the ten main operator blocks to four; measure final-use quality and runtime.',
                 hypothesis='Recent quality-only changes yield small or negative increments. A shallower student can trade training work for lower inference cost without changing allowed inputs or increasing ensemble count. Counter-explanation: removed depth carries essential temporal dynamics and cannot be recovered by distillation.',
                 historical_difference='C665 distilled correction/readout combinations; this compresses the whole ten-block shared backbone of the current C1014-1017 teacher, retaining width128 and all four interval heads. No external weights, generated flow or all81 teacher inside a held fold.',
                 comparison='Teacher versus separately trained four-block student. Teacher was fitted only on the same real training cases. Extra teacher-input distillation is explicitly counted; this is a whole-system comparison, not equal-compute causal attribution. Supervised-stage source/reflection order replays the teacher fit.',
                 scale='All three full fits:6000+12000,7259+14518,7457+14914 updates; all29/17/16 cases per actual ZIP twice. Conditional full81:9232+18464 and deployment validation.',
                 finish='Complete every declared fit, repeated whole evaluation and adoption decision, preserve models and images. Qualify accepted full81 ZIP through both dependencies/actual NPZ scorer/directC974 timing/full hashes before stock.',
                 next_action='After all current replay conditions finish, train the 4-block model on the full data with the teacher of the same fold. Check quality on all conditions, real speed and the submission path, then save.',
                 budget='Wait for current3 replay fits and any admitted stock continuation to finish; then one existing GPU job at a time, allocator4GiB/owned8GiB/host20GiB. Preserve all costs. No new paid resource, timeout or elapsed-time cancellation.',
                 training=dict(seed=old['training']['seed'],steps=warm+fine,warmup_steps=warm,finetune_steps=fine,retained_teacher_blocks=[0,4,5,9],
                               teacher_depth=10,student_depth=4,point_lr=1e-4,head_lr=8e-4,EMA=old['training']['EMA'],dropout=.03,observed_width=old['training']['observed_width'],
                               objective_source='scripts/inferred_score_objective.py',objective_sha256=sha(R/'scripts/inferred_score_objective.py'),
                               loss='Warm: relativeL2(student,teacher)+.1 normalized hidden MSE. Fine: empirical real composite +same distillation. Teacher always eval/frozen; no teacher in inference.'),
                 middle_step=warm+fine//2,
                 decision=dict(SPS_floor=-.15,point_minimum_delta={'rel_l2_score':-.03,'tke_score':-.25,'mvpe_score':-.05},
                               material_delta=.1,quality_floor=-.02,quality_CI90_lower_floor=-.05,
                               criterion='All3 same-teacher component floors, Time90 quality delta>=-.02 and CI90lower>=-.05. Full81 training-exposed same floors and quality>=-.02. Final stock additionally both warm round ratios/CIupper<=.65 and eachpair<=.70 vs actual C974; startup original<=1.10, full loader+NPZ/scorer+hash gates.',
                               runtime_target='Measured >=35% warm reduction vs actual PublicC974, not an inferred Public score.'),
                 runtime_change='Generalize old12-block assertion to checkpoint depth plus multiscale blocks. Per-block guards and runtime arithmetic unchanged; actual predict-NPZ-scorer preflight before learning.',
                 actual_submission=False,final_selection=False,time_based_termination=False)
        observer=(parent/'observe.py').read_text()
        old_accept="accepted=safe if m['full81'] else (safe and inferred['delta']>=.1 and inferred['CI90'][0]>0)"
        assert observer.count(old_accept)==1
        observer=observer.replace(old_accept,"accepted=safe and inferred['delta']>=m['decision']['quality_floor'] and (m['full81'] or inferred['CI90'][0]>=m['decision']['quality_CI90_lower_floor'])")
        observer=observer.replace('backbone dropout3% / matched composite fit / fixed midpoint-final EMA average','same-fold teacher depth10 vs distilled depth4 / extra training / fixed EMA average')
        observer=observer.replace('backbone dropout 3% + inferred composite loss','10-block teacher -> distilled 4-block')
        observer=observer.replace('Real-ZIP comparison with the same initial values, source/reflection streams, all updates and mid/final EMA averaging, with backbone dropout 3% enabled during training. Official aggregation, same cases.','All-case comparison of the trained teacher of the same fold and the student compressed to 4 blocks. Includes the extra distillation compute; not an equal-budget comparison. Speed is judged by a separate real-ZIP check.')
        observer=observer.replace('Same initial values, LR, reflection sample stream and all updates checked; inferred composite loss kept; training-time dropout 3% enabled at 12 places in the backbone, disabled at inference. The 50% merge of the source candidate rebuilt bit-exactly and the real ZIP evaluated on all cases.','Teacher fold boundaries and real checkpoints kept; 4-block initialization, all updates and the fixed EMA merge checked. Real ZIP without the teacher at inference evaluated on all cases.')
        (d/'observe.py').write_text(observer);compile(observer,str(d/'observe.py'),'exec')
        timing=(R/'runs/C1019R_simulation_replay/fast_runtime_timing_import_preflight.py').read_text()
        (d/'prepare_timing.py').write_text(timing)
        pins=[Path(__file__),R/'scripts/train_depth_distillation.py',d/'quality.py',d/'observe.py',d/'prepare_timing.py',d/'reflection_transform.py',d/'inputs_index.json',
              R/'scripts/preflight_depth_submission.py',parent/'plan.json',parent/'confirmation/report.json',R/teacher['path'],R/teacher['checkpoint'],R/m['evaluation_ref'],R/'scripts/inferred_score_objective.py']
        pins += [R/'src/realpde'/name for name in ['model.py','data.py','temporal_start_dataset.py','metrics.py','identity_interval_hardening.py','hidden_sps_interval.py','sps_radius_capacity.py','fixed_sps_head_training.py']]
        pins += [R/'scripts'/name for name in ['qualify_learning_candidate.py','submission_release_gate.py','experiment_records.py']]
        pins += [parent/'training_candidate'/f'resume_{fine}.pt']
        pins+=list(p for p in rt.rglob('*') if p.is_file())
        m['source_pins']={rel(p):sha(p) for p in pins}
        write(d/'plan.json',m);write(d/'base_quality.json',q)
    write(D/'cycle_plan.json',dict(trials=[rel(folder(c)) for c,_ in CONFIG[:3]],full81=rel(folder('C1031H')),primary_preserved=rel(PREVIOUS),
                                  criterion=read(D/'plan.json')['decision'],no_model_signals=True,actual_submission=False,time_based_termination=False))
    write(D/'cost_live.json',dict(commands=[],CPU_preparation_reserve_seconds=1200.,own_actual_command_seconds=0,user_time_cap_seconds=None,new_paid_spend=0))


def costs():
    prior=read(D/'prior_binding.json')['path'];commands={};reserves={}
    for p in [D/'cost_live.json',R/'runs/C1024R_depth_distillation/cost_live.json',R/'runs/C1031Q_deployment_validation/budget_live.json']:
        if not p.exists():continue
        v=read(p);reserves[rel(p)]=v.get('CPU_preparation_reserve_seconds',0)
        for c in v['commands']:assert c['receipt'] not in commands;commands[c['receipt']]=c['seconds']
    base=read(R/prior)['cumulative_charged_seconds']
    return dict(prior_ref=prior,prior_sha256=sha(R/prior),prior_cumulative_seconds=base,commands=commands,reserves=reserves,
                cumulative_charged_seconds=base+sum(commands.values())+sum(reserves.values()),user_time_cap_seconds=None,new_paid_spend=0,time_based_termination=False)


def command(d,label,argv,charge=True):
    resources.D=D;resources.wait_resource();start=time.monotonic()
    with (d/(label+'.log')).open('x') as log:
        p=subprocess.Popen(argv,cwd=R,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(D/'status.json',dict(stage=label,candidate=read(d/'plan.json')['candidate'],directory=rel(d),pid=p.pid,at=time.time()));code=p.wait()
    receipt=d/(label+'_execution.json');write(receipt,dict(command=p.args,pid=p.pid,exit_code=code,actual_seconds=time.monotonic()-start,time_based_termination=False))
    if charge:
        v=read(D/'cost_live.json');v['commands'].append(dict(receipt=rel(receipt),seconds=read(receipt)['actual_seconds']));v['own_actual_command_seconds']=sum(c['seconds'] for c in v['commands']);write(D/'cost_live.json',v)
    register(dict(candidate=read(d/'plan.json')['candidate'],kind='diagnostic' if code==0 else 'failure',summary='Teacher distillation to 4 blocks '+label+' exit '+str(code),
                  issue='Starting or finishing training is not a performance gain; judge by quality on all conditions and real speed.',next_action=read(d/'plan.json')['next_action'],
                  critical_check=dict(risk='Teacher fold leakage, or counting unfinished updates or missing NPZ as success.',finding='Exit receipt saved; real checkpoint, evaluation and submission checks verified afterwards.',evidence=rel(receipt)),
                  evidence=[rel(receipt)],visual=False,is_best=False),d/(label+'_event.json'))
    assert code==0,(label,code)


def package(d):
    import torch
    torch.set_num_threads(2);m=read(d/'plan.json');t=d/'training_candidate';report=read(t/'report.json')
    assert report['status']=='COMPLETE' and report['teacher_unchanged'] and report['fine_source_and_reflection_stream_exact_to_teacher_fit']
    end=torch.load(t/f"resume_{m['training']['steps']}.pt",map_location='cpu',weights_only=False)
    middle=torch.load(t/f"resume_{m['middle_step']}.pt",map_location='cpu',weights_only=False)
    assert end['step']==report['steps_actual']==m['training']['steps'] and end['plan_sha256']==middle['plan_sha256']==sha(d/'plan.json')
    assert end['model_config']['depth']==middle['model_config']['depth']==4
    def average(a,b):
        assert a.keys()==b.keys()
        return {k:(a[k]*.5+b[k]*.5 if a[k].is_floating_point() or a[k].is_complex() else a[k].clone()) for k in a}
    point=average(middle['ema_point'],end['ema_point']);heads=average(middle['ema_heads'],end['ema_heads'])
    cp=d/(m['candidate']+'_deployed_checkpoint.pt');torch.save(dict(ema_point=point,ema_heads=heads,model_config=end['model_config'],step=end['step']),cp)
    teacher=m['teacher_archive'];fast=m['runtime_baseline'];members={};target=d/(m['candidate']+'_DEPTH4_RESEARCH.zip')
    assert sha(R/teacher['path'])==teacher['sha256'] and sha(R/fast['path'])==fast['sha256']
    with zipfile.ZipFile(R/teacher['path']) as old,zipfile.ZipFile(R/fast['path']) as runtime,zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in runtime.namelist():
            if name.startswith('training_recipe/'):continue
            data=runtime.read(name)
            if name=='submission.py':data=depth_runtime(data)
            if name in WEIGHTS:
                payload=torch.load(io.BytesIO(old.read(name)),map_location='cpu',weights_only=False);key=WEIGHTS[name]
                payload['state_dict']=point if key=='point' else {k[len(key)+1:]:v for k,v in heads.items() if k.startswith(key+'.')}
                if key=='point':payload['model_config']=end['model_config']
                b=io.BytesIO();torch.save(payload,b);data=b.getvalue()
            z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        for name in old.namelist():
            if name.startswith('training_recipe/'):
                data=old.read(name);z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        for name,data in [('training_recipe/depth_distillation.json',json.dumps(m,ensure_ascii=False,indent=2).encode()),('training_recipe/depth_distillation_train.py.txt',(R/'scripts/train_depth_distillation.py').read_bytes())]:
            assert name not in members;z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and sum(i.file_size for i in z.infolist())<256000000
        for name,key in WEIGHTS.items():
            payload=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False);expected=point if key=='point' else {k[len(key)+1:]:v for k,v in heads.items() if k.startswith(key+'.')}
            assert payload['state_dict'].keys()==expected.keys() and all(torch.equal(v,expected[k]) for k,v in payload['state_dict'].items())
    candidate=dict(path=rel(target),sha256=sha(target),runtime_members=members,checkpoint=rel(cp),checkpoint_sha256=sha(cp),all_five_deployed_states_exact=True,step=end['step'],new_training=True)
    write(d/'package_audit.json',dict(status='COMPLETE',teacher=teacher,candidate=candidate,teacher_depth=10,student_depth=4,all_weight_tensors_exact=True,executable_runtime_source=fast['path'],runtime_change='Only block-count assertion tracks checkpoint configuration, preserving per-block guards and arithmetic.',no_teacher_inference=True))
    q=read(d/'base_quality.json');q['archives']=dict(control=teacher,candidate=candidate)
    q['quality_source_pins']={rel(p):sha(p) for p in [d/'quality.py',d/'inputs_index.json',d/'package_audit.json',R/m['evaluation_ref'],R/'scripts/evaluate_submission_equivalent.py',R/'vendor/starting_kit_v9/scoring.py',R/'src/realpde/c628_evaluation.py']}
    write(d/'quality_plan.json',q)
    for rnd in ['first','confirmation']:
        f=d/rnd;f.mkdir();write(f/'manifest.json',q)
        for arm,a in q['archives'].items():
            rt=f/'runtimes'/arm;rt.mkdir(parents=True)
            with zipfile.ZipFile(R/a['path']) as z:z.extractall(rt)


def evaluate(d):
    command(d,'package',[PY,str(Path(__file__)),'package',rel(d)])
    spec=importlib.util.spec_from_file_location('depth_observer_'+d.name,d/'observe.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    for rnd,arms in [('first',['control','candidate']),('confirmation',['candidate','control'])]:
        for arm in arms:command(d,rnd+'_'+arm,[PY,'-I',str(d/'quality.py'),'--round',rnd,'--arm',arm])
        report=observer.summary(d,rnd)
    m=read(d/'plan.json');r=report['inferred_score'];q=read(d/'quality_plan.json')
    scope=read(d/'confirmation/observation_event.json')['scope']
    register(dict(candidate=m['candidate'],kind='performance',candidate_status='parked',summary=m['boundary']+': 4-block model vs the teacher of the same fold, all cases twice. Quality difference '+format(r['delta'],'+.6f')+', before speed evaluation.',
                  issue='Quality comparison at fixed time. Quality floor and measured time checked separately; held surfaces already used.',next_action=m['next_action'],
                  metric=dict(name='inferred_composite_fixed_time90',direction='max',evaluation_id=m['derived_evaluation_id'],evaluation_ref=m['derived_evaluation_ref'],baseline_id=m['baseline'],baseline=r['baseline'],candidate=r['candidate'],material_delta=.1,lower_bound=r['CI90'][0]),
                  valid=True,guardrails_pass=report['safe'],run_id=m['candidate']+'_depth4_confirmation_pid'+str(report['actual_pids']['candidate']),scope=scope,
                  critical_check=dict(risk='Mismatch between compressed weights and the evaluated ZIP; reuse of the full81 teacher across folds.',finding='Teacher of the same fold fixed. 4 blocks, fixed EMA, all 5 weights and the real ZIP checked; official components of all rows and repeated identical predictions confirmed.',evidence=rel(d/'package_audit.json')),
                  evidence=[rel(d/'confirmation/report.json'),rel(d/'package_audit.json')],artifacts=[q['archives']['candidate']['path']],package_stage='local_validated',visual=True,media=report['media'],is_best=False),d/'confirmation/whole_measurement_event.json')
    write(d/'complete.json',dict(status='COMPLETE',accepted=report['accepted'],inferred_score=r));return report


def qualify(d):
    m=read(d/'plan.json');q=R/'runs/C1031Q_deployment_validation';t=R/'runs/C1031T_public_runtime'
    prior=d/'qualification_prior_cost.json';write(prior,costs())
    command(d,'qualification',[PY,str(R/'scripts/qualify_learning_candidate.py'),'--source-run',rel(d),'--run',rel(q),'--prior-cost',rel(prior)],charge=False)
    assert read(q/'qualification_complete.json')['status']=='PASS'
    command(d,'timing',[PY,str(d/'prepare_timing.py'),'--source-run',rel(d),'--qualification',rel(q),'--out',rel(t)])
    timing=read(t/'measurement.json');ok=timing['accepted_timing'] and all(v['block_bootstrap_90'][1]<=.65 and max(v['process_pair_ratios'])<=.70 for v in timing['rounds'])
    archive=read(d/'quality_plan.json')['archives']['candidate'];proof=q/'deployment_gate_proof.json'
    check_proof(R,archive['path'],rel(proof))
    for item in read(d/'inputs_index.json'):
        a=read(d/'confirmation/quality/candidate'/(item['case']+'.json'))
        for rnd in ['pip_first','pip_confirmation','image_first','image_confirmation']:
            b=read(q/rnd/'quality/candidate'/(item['case']+'.json'))
            assert all(a[k]==b[k] for k in ['prediction_hashes','mean_errors','sps_score','input_sha256','target_sha256','window_starts'])
    mutation('check','--deep');dest=None
    if ok:
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_JST')
        dest=R/'submissions'/(stamp+'__READY__C1031H_FULL81.zip');assert not dest.exists() and len(dest.name)<=64
        shutil.copy2(R/archive['path'],dest);assert sha(dest)==archive['sha256'];check_proof(R,rel(dest),rel(proof))
    decision=d/'stock_decision.json';write(decision,dict(status='QUALIFIED_LOCAL_STOCK' if ok else 'QUALIFIED_RESEARCH_TIMING_FAILED',accepted=ok,ready_zip=rel(dest) if dest else None,
                                                        archive_sha256=archive['sha256'],deployment_gate=rel(proof),timing=rel(t/'measurement.json'),runtime_baseline='C974H',official_server_reproduced=False,Public_prediction=None,actual_submission=False))
    register(dict(candidate=m['candidate'],kind='delivery',candidate_status='ready' if ok else 'parked',summary='Teacher-distilled 4-block model: all conditions, full81, submission path and direct C974 speed comparison done. '+('Candidate passing the fixed criteria saved.' if ok else 'Saved for research (speed criterion not met).'),
                  issue='Held conditions already used; full81 is in-sample. Official dependencies run on the local host/driver.',next_action='Keep the best Public candidate C974 and the quality candidate C1018; compare the new candidate by measurement.',
                  critical_check=dict(risk='Hiding a quality drop or a submission-path failure behind the speed gain.',finding='Pre-registered quality floor, full81 reproduction, real NPZ scoring in 2 environments, 12 direct C974 timing pairs and all hashes checked.',evidence=rel(decision)),
                  evidence=[rel(decision),rel(proof),rel(t/'measurement.json'),rel(d/'confirmation/report.json')],deployment_gate=rel(proof),artifacts=[rel(dest)] if dest else [archive['path']],package_stage='local_validated',visual=True,media=read(d/'confirmation/report.json')['media'],is_best=False),d/'stock_event.json')


def main():
    if not D.exists():prepare()
    for cid,_ in CONFIG:assert all(sha(R/p)==h for p,h in read(folder(cid)/'plan.json')['source_pins'].items())
    for cid,_ in CONFIG:assert read(folder(cid)/'submission_preflight/report.json')['status']=='PASS'
    for cid,_ in CONFIG[:3]:mutation('start','--secondary','--spec',rel(folder(cid)/'plan.json'))
    while (read(PREVIOUS/'status.json').get('stage')!='COMPLETE' or read(PREVIOUS/'fast_stock_continuation_status.json').get('stage') not in {'COMPLETE','NOT_ADMITTED_NO_RUNTIME_RUN'}):
        write(D/'status.json',dict(stage='WAIT_REPLAY_COMPLETION',candidate='C1028B',at=time.time(),no_model_signal=True));time.sleep(15)
    latest=read(R/'.records/work.json')['active']['candidate'];prior=PREVIOUS/'combined_cost_final.json'
    if latest=='C1023H':prior=R/'runs/C1023R_fast_quality_stock/combined_cost_final.json'
    write(D/'prior_binding.json',dict(path=rel(prior),sha256=sha(prior)))
    state=read(R/'.records/work.json');old_status='ready' if state['candidates'][latest].get('status')=='ready' else 'parked'
    switch=D/'switch.json';write(switch,dict(previous=latest,old_status=old_status,new_manifest=rel(D/'plan.json'),reason='Declared replay learning/evaluations and its conditional stock continuation completed. Begin predeclared full same-fold teacher compression.',remaining_work='All3 full distillation/fine-tuning fits, allcase repeats, adoption and conditional full81 deployment/timing/stock.',resume_when='Replay evidence and checkpoints remain available; preserve all old Public and qualified stock.',resource_transfer='Prior family completed. One allocated GPU job at a time; all cumulative costs retained; no elapsed-time stop.',evidence=[rel(PREVIOUS/'cycle_summary.json'),rel(prior)]));mutation('switch','--spec',rel(switch))
    results=[]
    for cid,_ in CONFIG[:3]:
        d=folder(cid);command(d,'train_candidate',[PY,str(R/'scripts/train_depth_distillation.py'),'--run',rel(d)]);results.append(evaluate(d))
    accepted=all(r['accepted'] for r in results)
    write(D/'cycle_summary.json',dict(status='COMPLETE',accepted_for_full81=accepted,boundaries=[{k:r[k] for k in ['candidate','delta','inferred_score','safe','accepted']} for r in results],actual_submission=False,Public_prediction=None))
    if accepted:
        d=folder('C1031H');mutation('start','--secondary','--spec',rel(d/'plan.json'));command(d,'train_candidate',[PY,str(R/'scripts/train_depth_distillation.py'),'--run',rel(d)]);r=evaluate(d)
        if r['accepted']:qualify(d)
        else:write(d/'stock_decision.json',dict(status='RESEARCH_ONLY',accepted=False,reason='Full81 training-exposed safety failed.'))
    else:write(D/'stock_decision.json',dict(status='RESEARCH_ONLY',accepted=False,reason='All3 fixed quality floors not met.',protected=['C974H','C1018H']))
    write(D/'combined_cost_final.json',costs())
    register(dict(candidate='C1028B',kind='diagnostic',candidate_status='parked',summary='Teacher distillation to 4 blocks: full training, all-condition comparison, fixed decision and artifact saving done.',issue='Full81 is in-sample; the other conditions were already used.',next_action='Keep the measured quality and time and the protected candidates for the next decision.',critical_check=dict(risk='Abandoning the main trial unfinished for time or condition reasons.',finding='All 3 conditions and the conditional full81/submission checks completed against fixed criteria; full cost saved.',evidence=rel(D/'cycle_summary.json')),evidence=[rel(D/'cycle_summary.json'),rel(D/'combined_cost_final.json')],visual=False,is_best=False),D/'completion_event.json')
    write(D/'status.json',dict(stage='COMPLETE',candidate='C1028B',at=time.time()));mutation('check')


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='package':package(R/sys.argv[2])
        else:main()
    except BaseException as error:
        if D.exists():write(D/'completion_failure.json',dict(error=repr(error),at=time.time(),no_other_model_signal=True))
        raise
