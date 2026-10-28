"""Jointly adapt the actual point backbone and all four deployed interval heads."""
import argparse,copy,gc,hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(seconds,arm):
 import torch
 from torch.utils.checkpoint import checkpoint
 from realpde import identity_interval_hardening as M
 from realpde.data import discover_cases
 from realpde.temporal_start_dataset import ReleasedTemporalStartDataset
 from realpde.fixed_sps_head_training import TrainingBudget
 from realpde.hidden_sps_interval import load_hidden_residual
 from realpde.sps_radius_capacity import load_expanded_head
 from realpde.model import OperatorBlock
 from realpde.metrics import relative_l2,tke_relative_l2,profile_relative_l2
 class Budget(TrainingBudget):
  def check(self):
   v=self.snapshot()
   for k,limit in [('peak_host_bytes',20*1024**3),('peak_device_bytes',8*1024**3)]:
    if v[k]>limit:raise RuntimeError('RESOURCE_CAP '+k)
   return v
 m=read(OUT/(arm+'_plan.json'));assert all(sha(ROOT/p)==h for p,h in m['source_pins'].items());dest=OUT/('training_'+arm);dest.mkdir();budget=Budget(dest,seconds,seconds)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(m['training']['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);device=torch.device('cuda');budget.gpu_started=time.monotonic()
 rt=ROOT/m['runtime'];spec=importlib.util.spec_from_file_location('c716g_point',rt/'factorized_model.py');F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
 import inspect,textwrap
 detached_forward=F.FactorizedFlowOperator.forward;forward_code=textwrap.dedent(inspect.getsource(detached_forward));assert forward_code.count('hidden.detach()')==2
 namespace=dict(F.__dict__);exec(forward_code.replace('hidden.detach()','hidden'),namespace);F.FactorizedFlowOperator.forward=namespace['forward']
 cp=torch.load(rt/'model.pt',map_location='cpu',weights_only=False);point=F.FactorizedFlowOperator(F.ModelConfig.from_dict(cp['model_config'])).to(device);point.load_state_dict(cp['state_dict']);point.eval().requires_grad_(True)
 keys={'center':'center_head.pt','radius':'radius_head.pt','hidden':'hidden_head.pt','observed':'observed_center_head.pt'}
 heads=torch.nn.ModuleDict({k:M.load_head(rt/f,device) if k=='center' else load_expanded_head(rt/f,device) if k=='radius' else load_hidden_residual(rt/f,device) for k,f in keys.items()}).eval().requires_grad_(True)
 ema_point=copy.deepcopy(point).eval().requires_grad_(False);ema_heads=copy.deepcopy(heads).eval().requires_grad_(False);initial={'point':M.state_hash(point),**{k:M.state_hash(v) for k,v in heads.items()}}
 original=OperatorBlock.forward
 def block(self,x):
  if torch.is_grad_enabled() and x.requires_grad:return checkpoint(lambda a:original(self,a),x,use_reentrant=False,preserve_rng_state=True)
  return original(self,x)
 OperatorBlock.forward=original
 def forward(x):
  hidden=[];handle=point.final_norm.register_forward_hook(lambda mod,inp,out:hidden.append(out))
  try:p=point(x)
  finally:handle.remove()
  assert len(hidden)==1;h=hidden[0];diagnostics,scale=M.flow_diagnostics(x,p);shift,_=heads['center'](diagnostics);_,radius=heads['radius'](diagnostics);ds,dl=heads['hidden'](h);raw=((x[...,:2]-x[...,:2].mean(1,keepdim=True))/scale).permute(0,1,4,2,3).reshape(-1,40,32,64);obs,_=heads['observed'](torch.cat((h,raw),1));return p,shift+ds+obs,radius*dl.exp(),scale,h
 try:
  assert len(m['train_cases'])==81 and m['eval_cases']==[] and '7575_0.h5' not in m['train_cases']
  cases=[c for c in discover_cases(str(ROOT/'data/extracted/train_real'),exclude_files=['7575_0.h5']) if c.path.name in m['train_cases']];assert [c.path.name for c in cases]==m['train_cases'] and len(cases)==81 and not set(m['train_cases'])&set(m['eval_cases']);data=ReleasedTemporalStartDataset(cases,stride=m['sampling']['stride'],budget=budget);data.windows=[(i,s) for i,c in enumerate(cases) for s in range(0,c.frames-m['sampling']['support_frames']+1,m['sampling']['stride'])];assert len(data)==m['sampling']['windows']
  from realpde.data import FlowWindowDataset
  checked=[]
  for ci in [0,len(cases)-1]:
   disk=FlowWindowDataset([cases[ci]],stride=1)
   valid_starts=[s for i,s in data.windows if i==ci]
   for local in [valid_starts[0],valid_starts[len(valid_starts)//2],valid_starts[-1]]:
    pos=data.windows.index((ci,local));xx,yy,_=data[pos];dx,dy=disk[local];assert torch.equal(xx,dx) and torch.equal(yy,dy);checked.append({'case':cases[ci].path.name,'start':local,'last_input':local+19,'first_target':local+20,'last_target':local+39})
  write(dest/'sampling_checks.json',{'all_cases':81,'windows':len(data),'stride':m['sampling']['stride'],'support_frames':m['sampling']['support_frames'],'first_middle_last_windows_bitexact_official_dataset':checked,'paired_sampling':m['sampling'],'new_case_groups':0,'new_independent_cases':0,'synthetic_data':False,'eval_case_overlap':False})
  write(dest/'job.json',{'pid':os.getpid(),'train_cases':m['train_cases'],'sample_ids':[[cases[i].path.name,s] for i,s in data.windows],'train_h5_sha256':{c.path.name:sha(c.path) for c in cases},'outer_target_reads':False,'initial_states':initial,'steps_planned':m['training']['steps'],'source_plan_sha256':sha(OUT/(arm+'_plan.json')),'dropout_disabled_eval_mode':True})
  opt=torch.optim.AdamW([{'params':point.parameters(),'lr':m['training']['point_lr']},{'params':heads.parameters(),'lr':m['training']['head_lr']}],weight_decay=1e-4);steps=m['training']['steps'];schedule_steps=m['training']['schedule_steps'];scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:.01+.99*.5*(1+__import__('math').cos(__import__('math').pi*t/schedule_steps)));rng=torch.Generator().manual_seed(m['training']['seed']+1);digest=hashlib.sha256();seen=set();last_step=0;started=time.monotonic();first_checks=None;order=torch.randperm(len(data),generator=rng);cursor=0
  with (dest/'curve.jsonl').open('x') as log:
   for step in range(1,steps+1):
    budget.check()
    if cursor>=len(data):order=torch.randperm(len(data),generator=rng);cursor=0
    ix=order[cursor:cursor+8];cursor+=len(ix);seen.update(ix.tolist());digest.update(ix.numpy().tobytes());items=[data[i] for i in ix.tolist()];x=torch.stack([v[0] for v in items]).to(device);target=torch.stack([v[1] for v in items]).to(device);opt.zero_grad(set_to_none=True);pred,shift,radius,scale,h=forward(x)
    l2=relative_l2(pred,target);tke=tke_relative_l2(pred,target);mvpe=profile_relative_l2(pred,target);early=relative_l2(pred[:,:3],target[:,:3]);factor=M.sample_factor(pred,target);err=(target[...,:2]-pred[...,:2])/scale;mask=target[...,:2]!=0;tau=.04*10**(-(step-1)/(schedule_steps-1));reward=(factor[:,None,None,None,None]*torch.sigmoid((radius-(err-shift).abs())/tau)*torch.exp(-2*radius*scale/M.SIGMA_GLOBAL))[mask].mean();pc=pred[...,:2]-pred[...,:2].mean(1,keepdim=True);tc=target[...,:2]-target[...,:2].mean(1,keepdim=True);pn=torch.linalg.vector_norm(pc,dim=(1,4),keepdim=True);tn=torch.linalg.vector_norm(tc,dim=(1,4),keepdim=True);wave=relative_l2(pc*(tn/pn.clamp_min(1e-8)),tc);late=relative_l2(pred[:,10:],target[:,10:]);loss=l2+1.*tke+.1*mvpe+.25*early-m['training']['sps_weight']*reward+m['training']['wave_weight']*wave+m['training']['late_weight']*late
    if step==1:
     probe=torch.autograd.grad(reward,point.lift.weight,retain_graph=True)[0];energy_probe=torch.autograd.grad(tke,point.lift.weight,retain_graph=True)[0];assert float(energy_probe.norm())>1e-4
     with torch.no_grad():original_pred=detached_forward(point,x);assert torch.equal(original_pred,pred.detach())
     wave_probe=torch.autograd.grad(wave,point.lift.weight,retain_graph=True)[0];first_checks={'TKE_to_lift_grad_norm':float(energy_probe.norm()),'initial_forward_bitexact':True,'two_training_detaches_removed':True,'waveform_to_lift_grad_norm':float(wave_probe.norm()),'SPS_to_lift_grad_norm':float(probe.norm()),'hidden_requires_grad':h.requires_grad,'factor_requires_grad':factor.requires_grad,'prediction_requires_grad':pred.requires_grad};assert first_checks['SPS_to_lift_grad_norm']>0 and h.requires_grad

     all_parameters=tuple(point.parameters())+tuple(heads.parameters());fast_grads=torch.autograd.grad(loss,all_parameters,retain_graph=True,allow_unused=True)
     OperatorBlock.forward=block
     pp,ss,rr,sc,hh=forward(x)
     pred_fast,shift_fast,radius_fast,scale_fast,h_fast=pred,shift,radius,scale,h
     pred,shift,radius,scale,h=pp,ss,rr,sc,hh
     l2=relative_l2(pred,target);tke=tke_relative_l2(pred,target);mvpe=profile_relative_l2(pred,target);early=relative_l2(pred[:,:3],target[:,:3]);factor=M.sample_factor(pred,target);err=(target[...,:2]-pred[...,:2])/scale;mask=target[...,:2]!=0;tau=.04*10**(-(step-1)/(schedule_steps-1));reward=(factor[:,None,None,None,None]*torch.sigmoid((radius-(err-shift).abs())/tau)*torch.exp(-2*radius*scale/M.SIGMA_GLOBAL))[mask].mean();pc=pred[...,:2]-pred[...,:2].mean(1,keepdim=True);tc=target[...,:2]-target[...,:2].mean(1,keepdim=True);pn=torch.linalg.vector_norm(pc,dim=(1,4),keepdim=True);tn=torch.linalg.vector_norm(tc,dim=(1,4),keepdim=True);wave=relative_l2(pc*(tn/pn.clamp_min(1e-8)),tc);late=relative_l2(pred[:,10:],target[:,10:]);loss=l2+1.*tke+.1*mvpe+.25*early-m['training']['sps_weight']*reward+m['training']['wave_weight']*wave+m['training']['late_weight']*late
     reference_loss=loss;reference_grads=torch.autograd.grad(reference_loss,all_parameters,allow_unused=True)
     assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(fast_grads,reference_grads))
     assert torch.equal(pred,pred_fast)
     OperatorBlock.forward=original
     pred,shift,radius,scale,h=pred_fast,shift_fast,radius_fast,scale_fast,h_fast
     l2=relative_l2(pred,target);tke=tke_relative_l2(pred,target);mvpe=profile_relative_l2(pred,target);early=relative_l2(pred[:,:3],target[:,:3]);factor=M.sample_factor(pred,target);err=(target[...,:2]-pred[...,:2])/scale;mask=target[...,:2]!=0;tau=.04*10**(-(step-1)/(schedule_steps-1));reward=(factor[:,None,None,None,None]*torch.sigmoid((radius-(err-shift).abs())/tau)*torch.exp(-2*radius*scale/M.SIGMA_GLOBAL))[mask].mean();pc=pred[...,:2]-pred[...,:2].mean(1,keepdim=True);tc=target[...,:2]-target[...,:2].mean(1,keepdim=True);pn=torch.linalg.vector_norm(pc,dim=(1,4),keepdim=True);tn=torch.linalg.vector_norm(tc,dim=(1,4),keepdim=True);wave=relative_l2(pc*(tn/pn.clamp_min(1e-8)),tc);late=relative_l2(pred[:,10:],target[:,10:]);loss=l2+1.*tke+.1*mvpe+.25*early-m['training']['sps_weight']*reward+m['training']['wave_weight']*wave+m['training']['late_weight']*late
     first_checks['checkpoint_recompute_disabled_gradient_bitexact']=True;first_checks['checkpoint_recompute_disabled_forward_bitexact']=True;first_checks['all_point_and_heads_gradients_checked']=len(all_parameters)
     del pp,ss,rr,sc,hh,fast_grads,reference_grads,reference_loss
    assert torch.isfinite(loss);loss.backward();gp=torch.nn.utils.clip_grad_norm_(point.parameters(),1.,error_if_nonfinite=True);gh=torch.nn.utils.clip_grad_norm_(heads.parameters(),1.,error_if_nonfinite=True);assert gp>0 and gh>0;opt.step();scheduler.step();last_step=step
    with torch.no_grad():
     for dst,src in [(ema_point,point),(ema_heads,heads)]:
      for a,b in zip(dst.parameters(),src.parameters()):a.lerp_(b,1-m['training']['EMA'])
      for a,b in zip(dst.buffers(),src.buffers()):a.copy_(b)
    if step==1 or step%100==0:
     row={'step':step,'loss':float(loss.detach()),'rel_l2':float(l2.detach()),'tke':float(tke.detach()),'mvpe':float(mvpe.detach()),'early3':float(early.detach()),'energy_normalized_phase':float(wave.detach()),'late10':float(late.detach()),'soft_SPS_reward':float(reward.detach()),'tau':tau,'point_grad':float(gp),'heads_grad':float(gh),'lr':scheduler.get_last_lr(),'training_seconds':time.monotonic()-started};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);write(dest/'heartbeat.json',dict(step=step,**budget.check()))
    if step in {3000,6000,steps}:torch.save({'step':step,'point':point.state_dict(),'heads':heads.state_dict(),'ema_point':ema_point.state_dict(),'ema_heads':ema_heads.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.get_state(),'sampler_order':order,'sampler_cursor':cursor,'unique_sampled_windows':len(seen),'batch_digest':digest.hexdigest(),'plan_sha256':sha(OUT/(arm+'_plan.json'))},dest/('resume_'+str(step)+'.pt'))
  assert {cases[data.windows[i][0]].path.name for i in seen}==set(m['train_cases'])
  assert last_step==steps and scheduler.last_epoch==steps and len(seen)==min(len(data),steps*8);endpoints={}
  for endpoint,pm,hm in [('ema',ema_point,ema_heads),('live',point,heads)]:
   folder=dest/endpoint;folder.mkdir();parts={};payload=torch.load(rt/'model.pt',map_location='cpu',weights_only=False);payload['state_dict']={k:v.detach().cpu() for k,v in pm.state_dict().items()};payload['C758']={'steps_actual':last_step,'endpoint':endpoint,'source_plan_sha256':sha(OUT/(arm+'_plan.json'))};torch.save(payload,folder/'model.pt')
   parts['model.pt']={'path':str((folder/'model.pt').relative_to(ROOT)),'sha256':sha(folder/'model.pt'),'state_hash':M.state_hash(pm)}
   for k,f in keys.items():
    payload=torch.load(rt/f,map_location='cpu',weights_only=False);payload['state_dict']={n:v.detach().cpu() for n,v in hm[k].state_dict().items()};payload['C758']={'steps_actual':last_step,'endpoint':endpoint};torch.save(payload,folder/f);parts[f]={'path':str((folder/f).relative_to(ROOT)),'sha256':sha(folder/f),'state_hash':M.state_hash(hm[k])}
   assert parts['model.pt']['state_hash']!=initial['point'];r={'status':'COMPLETE','candidate':m['arms'][endpoint]['candidate'],'steps_actual':last_step,'train_cases':81,'train_windows':len(data),'unique_sampled_windows':len(seen),'endpoint':endpoint,'endpoints':parts,'batch_digest':digest.hexdigest(),'initial_states':initial,'outer_target_reads':False,'runtime_code_unchanged':True};write(folder/'report.json',r);endpoints[endpoint]=r
  assert not point.final_norm._forward_hooks;assert all(sha(ROOT/p)==h for p,h in m['source_pins'].items());write(dest/'report.json',{'status':'COMPLETE','arms':endpoints,'steps_actual':last_step,'gradient_checks':first_checks,'training_loop_seconds':time.monotonic()-started,'budget':budget.check(),'selection':'Fixed data-exposure-normalized endpoint, cosine/tau horizon and EMA from plan; not validation-selected. Verified C728 full81 parent; all81 training-exposed, not heldout. LIVE not evaluated or selected.','runtime_architecture_unchanged':True,'activation_checkpointing':'disabled, full gradients bitexact against checkpoint recompute at first real batch'})
 except BaseException as e:write(dest/'failure.json',{'error':repr(e),'budget':budget.snapshot()});raise
 finally:OperatorBlock.forward=original;write(dest/'budget.json',budget.snapshot())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);p.add_argument('--arm',choices=['previous','candidate'],required=True);a=p.parse_args();main(a.seconds,a.arm)
