"""Sim-focus (fixed emphasis ratio) variant of train_simati.py: imports runs/SIMALL/sim_replay_focus.py (training.sim_focus_fraction). Otherwise identical.
SIMATI: SIMAT trainer + interval distillation (training.interval_kd): the reflection-averaged teacher interval centre
(physical units) and log half-width are distilled into the student heads alongside the point, with the real-label SPS
learning kept. Teacher heads = frozen copy of the initial heads (the teacher runtime heads). Otherwise identical to SIMAT.
Original SIMAT docstring: SIMA trainer with an optional reflection-averaged KD teacher (training.teacher_tta): point KD target = mean of T(x) and
R^-1 T(Rx) (the verified test-time average), so the deployed single-pass student can absorb the TTA gain. Everything else
identical to SIMA (teacher, stream, simulation replay, steps, EMA averaging)."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,inspect,json,os,sys,textwrap,time
R=Path(__file__).resolve().parents[2]  # this copy lives in runs/<dir>/, the original in scripts/
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from experiment_records import read,write,sha,rel



def phase_cadence(data,index,phase,cad):
    """Exact released subsampling in space (2x2 phase) and time (spacing 1 or 2); no generated values."""
    import torch
    assert 0<=phase<4 and cad in (0,1)
    ci,start=data.windows[index];dy,dx=divmod(phase,2)
    assert cad==0 or start+79<=data.flows[ci].shape[0]
    sp=cad+1;uv=data.flows[ci][start:start+40*sp:sp,dy::2,dx::2]
    assert uv.shape==(40,32,64,2),uv.shape
    native=torch.cat((uv,torch.zeros_like(uv[...,:1])),dim=-1)
    return native[:20],native[20:],torch.tensor(index,dtype=torch.long)

def main(directory,preflight=False):
    import torch
    from realpde import identity_interval_hardening as M
    from realpde.data import discover_cases,FlowWindowDataset
    from realpde.temporal_start_dataset import ReleasedTemporalStartDataset
    from realpde.hidden_sps_interval import load_hidden_residual
    from realpde.sps_radius_capacity import load_expanded_head
    from realpde.metrics import relative_l2,tke_relative_l2,profile_relative_l2
    from realpde.fixed_sps_head_training import TrainingBudget
    from inferred_score_objective import training_loss
    class Budget(TrainingBudget):
        def check(self):
            v=self.snapshot()
            for k,limit in [('peak_host_bytes',20*1024**3),('peak_device_bytes',8*1024**3)]:
                if v[k]>limit:raise RuntimeError('RESOURCE_CAP '+k)
            return v
    d=(R/directory).resolve();m=read(d/'plan.json');dest=d/('cpu_preflight' if preflight else 'training_candidate')
    assert not dest.exists();dest.mkdir()
    assert all(sha(R/p)==h for p,h in m['source_pins'].items())
    budget=Budget(dest,0,0);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(m['training']['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not preflight:torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    device=torch.device('cpu' if preflight else 'cuda');started=time.monotonic()
    if not preflight:budget.gpu_started=time.monotonic()
    rt=R/m['runtime'];spec=importlib.util.spec_from_file_location('depth_teacher_factorized',rt/'factorized_model.py')
    F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
    source=textwrap.dedent(inspect.getsource(F.FactorizedFlowOperator.forward));assert source.count('hidden.detach()')==2
    ns=dict(F.__dict__);exec(source.replace('hidden.detach()','hidden'),ns);F.FactorizedFlowOperator.forward=ns['forward']
    cp=torch.load(rt/'model.pt',map_location='cpu',weights_only=False)
    teacher=F.FactorizedFlowOperator(F.ModelConfig.from_dict(cp['model_config'])).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(cp['state_dict']);teacher_hash=M.state_hash(teacher)
    config=dict(cp['model_config']);assert config['depth']==10;SD=int(m['training'].get('student_depth',4));config['depth']=SD
    if m['training'].get('physics_features'):config['use_physics_features']=True
    if m['training'].get('student_dropout') is not None:config['dropout']=float(m['training']['student_dropout'])
    point=F.FactorizedFlowOperator(F.ModelConfig.from_dict(config)).to(device).train().requires_grad_(True)
    kept=m['training']['retained_teacher_blocks'];assert len(kept)==SD and kept==sorted(set(kept)) and max(kept)<10
    initial={}
    for name in point.state_dict():
        old=name
        if name.startswith('blocks.'):
            items=name.split('.');items[1]=str(kept[int(items[1])]);old='.'.join(items)
        w=cp['state_dict'][old]
        if name=='lift.weight' and w.shape!=point.state_dict()[name].shape:   # zero-initialised extra physics inputs keep the function unchanged
            extra=point.state_dict()[name].shape[1]-w.shape[1];assert extra==9;w=torch.cat((w,torch.zeros(w.shape[0],extra,*w.shape[2:],dtype=w.dtype)),1)
        initial[name]=w
    point.load_state_dict(initial);del cp,initial
    files={'center':'center_head.pt','radius':'radius_head.pt','hidden':'hidden_head.pt','observed':'observed_center_head.pt'}
    heads=torch.nn.ModuleDict({k:M.load_head(rt/f,device) if k=='center' else load_expanded_head(rt/f,device) if k=='radius' else load_hidden_residual(rt/f,device) for k,f in files.items()}).eval().requires_grad_(True)
    teacher_heads=copy.deepcopy(heads).eval().requires_grad_(False);IKD=[0.0]   # SIMATI: frozen teacher heads (initial = teacher runtime heads)
    initial_states={'point':M.state_hash(point),'heads':M.state_hash(heads)}
    dropout=[v for v in point.modules() if isinstance(v,torch.nn.Dropout)]
    assert len(dropout)==SD+2 and all(abs(v.p-config['dropout'])<1e-12 and v.training for v in dropout)
    HR=bool(m['training'].get('head_refit'))
    SIMR=None
    if m['training'].get('sim_replay') and not preflight:
        sys.path.insert(0,str(R/'runs/SIMALL'));sys.path.insert(0,str(R/'scripts'))
        from sim_replay_focus2 import SimulationReplay as _SR   # SIMSAT2D4M: fixed per-window focus emphasis (sim_focus_ratio)
        SIMR=_SR(R,m,dest,budget,device,policy=m['training']['sim_replay'])
    if HR:point.eval().requires_grad_(False)   # frozen deployed point, dropout off: heads are refit to the deployed residuals
    transform=importlib.util.spec_from_file_location('depth_reflection',R/m['reflection_source'])
    reflection=importlib.util.module_from_spec(transform);transform.loader.exec_module(reflection)
    def forward(model,x):
        hidden=[];hook=model.final_norm.register_forward_hook(lambda mod,inp,out:hidden.append(out))
        try:p=model(x)
        finally:hook.remove()
        return p,hidden[0]
    def interval_with(hd,x,p,h):
        diag,scale=M.flow_diagnostics(x,p);shift,_=hd['center'](diag);_,radius=hd['radius'](diag)
        ds,dl=hd['hidden'](h);raw=((x[...,:2]-x[...,:2].mean(1,keepdim=True))/scale).permute(0,1,4,2,3).reshape(-1,40,32,64)
        obs,odl=hd['observed'](torch.cat((h,raw),1));radius=radius*dl.exp()
        if m['training']['observed_width']:radius=radius*odl.exp()
        return shift+ds+obs,radius,scale
    def interval(x,p,h):
        diag,scale=M.flow_diagnostics(x,p);shift,_=heads['center'](diag);_,radius=heads['radius'](diag)
        ds,dl=heads['hidden'](h);raw=((x[...,:2]-x[...,:2].mean(1,keepdim=True))/scale).permute(0,1,4,2,3).reshape(-1,40,32,64)
        obs,odl=heads['observed'](torch.cat((h,raw),1));radius=radius*dl.exp()
        if m['training']['observed_width']:radius=radius*odl.exp()
        return shift+ds+obs,radius,scale
    def losses(native,target,selected,phase):
        x=torch.where(selected[:,None,None,None,None],reflection.reflect_velocity_fields(native),native)
        pred,h=forward(point,x)
        if HR:tp,th=pred.detach(),h.detach()   # frozen point equals the deployed teacher; skip the redundant teacher pass
        else:
            with torch.no_grad():
                tp,th=forward(teacher,x)
                if m['training'].get('teacher_tta'):   # reflection-averaged teacher point (TTA target); hidden-feature KD stays on the native view
                    tp_native=tp;tr,thr=forward(teacher,reflection.reflect_velocity_fields(x));tp=0.5*(tp+reflection.reflect_velocity_fields(tr))
        point_kd=relative_l2(pred,tp);feature_kd=(h-th).square().mean()/th.square().mean().clamp_min(1e-8)
        real_loss=pred.new_zeros(());reward=pred.new_zeros(())
        if phase=='finetune':
            shift,radius,scale=interval(x,pred,h)
            if m['training'].get('interval_kd'):   # reflection-averaged teacher interval (centre in physical units, log half-width), x geometry
                with torch.no_grad():
                    s1,r1,sc1=interval_with(teacher_heads,x,tp_native,th);c1=tp_native[...,:2]+s1*sc1;w1=r1*sc1
                    xr=reflection.reflect_velocity_fields(x);s2,r2,sc2=interval_with(teacher_heads,xr,tr,thr);c2=tr[...,:2]+s2*sc2;w2=r2*sc2
                    sg=c1.new_tensor([1.,-1.]);cT=0.5*(c1+c2.flip(2)*sg);wT=0.5*(w1+w2.flip(2))
                cS=pred[...,:2]+shift*scale;wS=radius*scale
                IKD[0]=float(m['training'].get('interval_kd_center',1.))*relative_l2(cS,cT)+float(m['training'].get('interval_kd_radius',.1))*(torch.log(wS.clamp_min(1e-6))-torch.log(wT.clamp_min(1e-6))).abs().mean()
            pred,shift,radius,scale=reflection.restore_native_geometry(pred,shift,radius,scale,native,selected)
            factor=M.sample_factor(pred,target);err=(target[...,:2]-pred[...,:2])/scale;mask=target[...,:2]!=0
            reward=(factor[:,None,None,None,None]*torch.sigmoid((radius-(err-shift).abs())/.01)*torch.exp(-2*radius*scale/M.SIGMA_GLOBAL))[mask].mean()
            real_loss=training_loss(relative_l2(pred,target),tke_relative_l2(pred,target),profile_relative_l2(pred,target),reward)
        KDW=float(m['training'].get('kd_weight',1.0))
        sim=SIMR.loss(point,heads) if (SIMR is not None and phase=='finetune') else 0.0
        ikd=IKD[0];IKD[0]=0.0
        return real_loss+KDW*(point_kd+.1*feature_kd)+sim+ikd,real_loss,point_kd,feature_kd,reward
    cases=[c for c in discover_cases(R/'data/extracted/train_real',exclude_files=['7575_0.h5']) if c.path.name in m['train_cases']]
    assert [c.path.name for c in cases]==m['train_cases'] and not set(m['train_cases'])&set(m['eval_cases'])
    parent=read(R/m['teacher_source']/'plan.json');assert parent['train_cases']==m['train_cases'] and parent['eval_cases']==m['eval_cases']
    if preflight:
        native,target=FlowWindowDataset([cases[0]],stride=1)[0];native=native.unsqueeze(0);target=target.unsqueeze(0)
        checks=[]
        for phase in (['finetune'] if HR else ['distillation','finetune']):
            point.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
            loss,real_loss,kd,hidden_loss,reward=losses(native,target,torch.tensor([True]),phase)
            assert torch.isfinite(loss);loss.backward()
            pg=sum(float(p.grad.norm()) for p in point.parameters() if p.grad is not None)
            hg={k:sum(float(p.grad.norm()) for p in v.parameters() if p.grad is not None) for k,v in heads.items()}
            assert (pg>0 or HR) and all(p.grad is None for p in teacher.parameters())
            assert all(v>0 for v in hg.values()) if phase=='finetune' else all(p.grad is None for p in heads.parameters())
            checks.append(dict(phase=phase,loss=float(loss.detach()),point_gradient=pg,head_gradients=hg,teacher_gradients_none=True))
        point.eval()
        with torch.no_grad():
            a=point(native);b=point(native)
            assert a.shape==native.shape and torch.equal(a,b) and torch.isfinite(a).all()
        assert M.state_hash(teacher)==teacher_hash and M.state_hash(point)==initial_states['point'] and M.state_hash(heads)==initial_states['heads']
        write(dest/'report.json',dict(status='PASS',scope='One training-only window; CPU graph/state/API preflight, no optimizer update and no performance conclusion.',
                                     sample=cases[0].path.name+'/start0',teacher_depth=10,student_depth=SD,student_parameters=sum(p.numel() for p in point.parameters()),
                                     teacher_parameters=sum(p.numel() for p in teacher.parameters()),checks=checks,eval_repeat_bitexact=True,
                                     actual_seconds=time.monotonic()-started,no_updates=True,no_GPU_model_work=True,source_plan_sha256=sha(d/'plan.json')))
        return
    from realpde.released_lattice_dataset import ReleasedLatticeDataset
    data=ReleasedLatticeDataset(cases,budget=budget)
    data.windows=[(i,s) for i,c in enumerate(cases) for s in range(c.frames-m['sampling']['support_frames']+1)]
    assert len(data)==m['sampling']['windows']
    checks=[]
    for ci in [0,len(cases)-1]:
        disk=FlowWindowDataset([cases[ci]],stride=1);starts=[s for i,s in data.windows if i==ci]
        for s in [starts[0],starts[len(starts)//2],starts[-1]]:
            xx,yy,_=data[data.windows.index((ci,s))];dx,dy=disk[s];assert torch.equal(xx,dx) and torch.equal(yy,dy)
            checks.append([cases[ci].path.name,s])
    write(dest/'job.json',dict(pid=os.getpid(),train_cases=m['train_cases'],teacher_source=m['teacher_source'],teacher_point_hash=teacher_hash,
                              initial_states=initial_states,train_h5_sha256={c.path.name:sha(c.path) for c in cases},steps_planned=m['training']['steps'],
                              teacher_eval_and_frozen=True,outer_target_reads=False,kept_blocks=kept,native_windows_checked=checks))
    warm=m['training']['warmup_steps'];fine=m['training']['finetune_steps'];total=warm+fine
    ema_point=copy.deepcopy(point).eval().requires_grad_(False);ema_heads=copy.deepcopy(heads).eval().requires_grad_(False)
    phases=[];global_step=0;first_checks={}
    with (dest/'curve.jsonl').open('x') as log:
        for phase,steps in [(ph,st) for ph,st in [('distillation',warm),('finetune',fine)] if st>0]:
            seed=m['training']['seed']+int(m['training'].get('stream_seed_offset',0));rng=torch.Generator().manual_seed(seed+(31 if phase=='distillation' else 1))
            ag=torch.Generator().manual_seed(seed+(32 if phase=='distillation' else 2))
            order=torch.randperm(len(data),generator=rng);cursor=0;seen=set();digest=hashlib.sha256();adigest=hashlib.sha256();sample_count=reflected=0;prng=torch.Generator().manual_seed(seed+(33 if phase=='distillation' else 3));crng=torch.Generator().manual_seed(seed+(34 if phase=='distillation' else 4));CAD_P=float(m['training']['cadence2_probability']);LP=list(m['training'].get('lattice_phases',[0,1,2,3]));assert LP and all(q in (0,1,2,3) for q in LP);lat_counts=[0]*4;cad_counts=[0,0];cad_fb=0
            opt=torch.optim.AdamW([{'params':point.parameters(),'lr':2e-4 if phase=='distillation' else 1e-4},{'params':heads.parameters(),'lr':8e-4}],weight_decay=1e-4)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:.01+.99*.5*(1+__import__('math').cos(__import__('math').pi*t/(2*steps))))
            if phase=='finetune':ema_point.load_state_dict(point.state_dict());ema_heads.load_state_dict(heads.state_dict())
            for step in range(1,steps+1):
                budget.check()
                if cursor>=len(data):order=torch.randperm(len(data),generator=rng);cursor=0
                ix=order[cursor:cursor+8];cursor+=len(ix);seen.update(ix.tolist());digest.update(ix.numpy().tobytes())
                phs=[LP[q%len(LP)] for q in torch.randint(4,(len(ix),),generator=prng).tolist()];req=(torch.rand(len(ix),generator=crng)<CAD_P).long().tolist()
                cads=[c if (c==0 or data.windows[i][1]+79<=data.flows[data.windows[i][0]].shape[0]) else 0 for i,c in zip(ix.tolist(),req)];cad_fb+=sum(1 for r,a in zip(req,cads) if r==1 and a==0)
                [lat_counts.__setitem__(q,lat_counts[q]+1) for q in phs];[cad_counts.__setitem__(c,cad_counts[c]+1) for c in cads]
                items=[phase_cadence(data,i,q,c) for i,q,c in zip(ix.tolist(),phs,cads)];native=torch.stack([v[0] for v in items]).to(device);target=torch.stack([v[1] for v in items]).to(device)
                flip=torch.rand(len(ix),generator=ag)<.5;adigest.update(flip.numpy().tobytes());sample_count+=len(ix);reflected+=int(flip.sum())
                selected=flip.to(device)
                opt.zero_grad(set_to_none=True)
                loss,real_loss,point_kd,feature_kd,reward=losses(native,target,selected,phase)
                assert torch.isfinite(loss);loss.backward()
                gp=torch.nn.utils.clip_grad_norm_(point.parameters(),1.,error_if_nonfinite=True)
                assert gp>0 or HR
                gh=torch.nn.utils.clip_grad_norm_(heads.parameters(),1.,error_if_nonfinite=True)
                if phase=='finetune':assert gh>0
                else:assert all(p.grad is None for p in heads.parameters())
                assert all(p.grad is None for p in teacher.parameters())
                opt.step();sched.step();global_step+=1
                with torch.no_grad():
                    for dst,src in [(ema_point,point),(ema_heads,heads)]:
                        for a,b in zip(dst.parameters(),src.parameters()):a.lerp_(b,1-m['training']['EMA'])
                        for a,b in zip(dst.buffers(),src.buffers()):a.copy_(b)
                if step==1:
                    first_checks[phase]=dict(point_gradient=float(gp),heads_gradient=float(gh),teacher_gradients_none=True)
                    write(dest/'first_intervention_checks.json',first_checks)
                if step==1 or step%100==0 or step==steps:
                    row=dict(step=global_step,phase=phase,phase_step=step,loss=float(loss.detach()),real_loss=float(real_loss.detach()),
                             point_distillation=float(point_kd.detach()),hidden_distillation=float(feature_kd.detach()),soft_SPS_reward=float(reward.detach()),
                             point_grad=float(gp),heads_grad=float(gh),training_seconds=time.monotonic()-started)
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                    write(dest/'heartbeat.json',dict(step=global_step,training_phase=phase,**budget.check()))
                if (phase=='distillation' and step==steps) or (phase=='finetune' and step in {fine//2,fine}):
                    torch.save(dict(step=global_step,phase=phase,phase_step=step,point=point.state_dict(),heads=heads.state_dict(),ema_point=ema_point.state_dict(),ema_heads=ema_heads.state_dict(),
                                    model_config=config,optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng.get_state(),augmentation_rng=ag.get_state(),
                                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_order=order,sampler_cursor=cursor,unique_sampled_windows=len(seen),
                                    batch_digest=digest.hexdigest(),augmentation_digest=adigest.hexdigest(),source_samples=sample_count,reflected_samples=reflected,
                                    plan_sha256=sha(d/'plan.json')),dest/f'resume_{global_step}.pt')
            phases.append(dict(phase=phase,steps=steps,unique_sampled_windows=len(seen),batch_digest=digest.hexdigest(),augmentation_digest=adigest.hexdigest(),source_samples=sample_count,lattice_phase_counts=lat_counts,cadence_counts=cad_counts,cadence_trailing_fallbacks=cad_fb,cadence2_probability=CAD_P))
            if phase=='finetune':assert len(seen)==len(data)
    assert global_step==total and M.state_hash(teacher)==teacher_hash
    assert all(sha(R/p)==h for p,h in m['source_pins'].items())
    original=torch.load(R/m['teacher_source']/'training_candidate'/f'resume_{fine}.pt',map_location='cpu',weights_only=False)
    REPLAY=bool(m['training'].get('replay_teacher_stream',True))
    assert (not REPLAY) or (phases[-1]['batch_digest']==original['batch_digest'] and phases[-1]['augmentation_digest']==original['augmentation_digest'])
    write(dest/'report.json',dict(status='COMPLETE',simulation_replay=(SIMR.report() if SIMR is not None else None),steps_actual=global_step,phases=phases,teacher_unchanged=True,
                                  fine_source_and_reflection_stream_exact_to_teacher_fit=REPLAY,initial_states=initial_states,
                                  teacher_source=m['teacher_source'],teacher_point_hash=teacher_hash,student_depth=SD,teacher_depth=10,
                                  gradient_checks=first_checks,budget=budget.check(),training_loop_seconds=time.monotonic()-started,
                                  export='Equal midpoint/final finetuning EMA, no validation checkpoint selection.',new_independent_data=False))
    write(dest/'budget.json',budget.snapshot())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--cpu-preflight',action='store_true');args=parser.parse_args()
    try:main(args.run,args.cpu_preflight)
    except BaseException as error:
        write(R/args.run/('cpu_preflight' if args.cpu_preflight else 'training_candidate')/'failure.json',dict(error=repr(error),at=time.time(),no_other_model_signal=True))
        raise
