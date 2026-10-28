"""Fixed 4,000-step hidden SPS head using explicit training cases and weights only.

No experiment imports, evaluation arrays, leaderboard, or calibration maps.
"""
from __future__ import annotations
import argparse,copy,datetime,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import identity_interval_hardening as method
from .data import discover_cases
from .fixed_sps_head_training import TrainingBudget,_write
from .hidden_sps_interval import HiddenIntervalResidual,predict_with_hidden,load_hidden_residual
from .sps_radius_capacity import load_expanded_head

def hidden_training_cache(frozen,dataset,device,output,budget,expected=None):
 cache={}
 for start in range(0,len(dataset),8):
  stop=min(start+8,len(dataset));items=[dataset[i] for i in range(start,stop)];x=torch.stack([i[0] for i in items]).to(device);target=torch.stack([i[1][...,:2] for i in items])
  with budget.cuda(),torch.inference_mode():
   p,h=predict_with_hidden(frozen['point'],x);d,scale=method.flow_diagnostics(x,p[...,:2]);shift,_=frozen['center'](d);_,rad=frozen['radius'](d)
   if start==0:assert torch.equal(p,frozen['point'](x).float()) and not frozen['point'].final_norm._forward_hooks
  values=dict(point=p[...,:2].cpu(),target=target,hidden=h.cpu(),scale=scale.cpu(),shift=shift.cpu(),radius=rad.cpu())
  if start==0:
   size=sum(v[0].numel()*4*len(dataset) for v in values.values());assert size<=8*1024**3;assert h.shape[1:]==(128,32,64)
  for k,v in values.items():
   assert v.dtype==torch.float32 and torch.isfinite(v).all()
   if k not in cache:cache[k]=torch.empty((len(dataset),*v.shape[1:]),dtype=torch.float32)
   cache[k][start:stop].copy_(v)
  if stop%256==0 or stop==len(dataset):budget.check();_write(output/'heartbeat.json',dict(stage='cache_'+str(stop),**budget.snapshot()))
 record=dict(rows=len(dataset),bytes=sum(v.numel()*v.element_size() for v in cache.values()),hashes={k:method.tensor_hash(v) for k,v in cache.items()},hook_point_bitexact=True,hook_removed=True)
 if expected is not None:
  assert record==expected,'hidden training cache does not reproduce'
 _write(output/'training_cache.json',record);return cache

def train_hidden_residual(frozen,cache,device,output,budget):
 ARMS=('aligned',)
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(2026081702);initial=HiddenIntervalResidual(128,20)
 models={a:copy.deepcopy(initial).to(device).train() for a in ARMS};assert len({method.state_hash(m) for m in models.values()})==1
 initial_hash=method.state_hash(initial);optim={a:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for a,m in models.items()};schedulers={a:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4000,eta_min=2e-6) for a,o in optim.items()};emas={a:method.ExponentialMovingAverage(m,.995) for a,m in models.items()};rng=torch.Generator().manual_seed(20260907586);digests={a:hashlib.sha256() for a in ARMS};history=[]
 with (output/'training_curve.jsonl').open('x') as f:
  for step in range(1,4001):
   ix=torch.randint(len(cache['point']),(8,),generator=rng);budget.check();row=dict(step=step,tau=.04*10**(-(step-1)/3999))
   with budget.cuda():
    batch={k:v.index_select(0,ix).to(device) for k,v in cache.items()};factor=method.sample_factor(batch['point'],batch['target']);remaining=(batch['target']-batch['point'])/batch['scale'];mask=batch['target']!=0
    for a,m in models.items():
     hidden=batch['hidden'] if a=='aligned' else torch.roll(batch['hidden'],shifts=(16,32),dims=(2,3))
     if step==1 and a=='spatial_roll':assert torch.equal(torch.roll(hidden,(-16,-32),(2,3)),batch['hidden'])
     optim[a].zero_grad(set_to_none=True);ds,dl=m(hidden);shift=batch['shift']+ds;rad=batch['radius']*dl.exp()
     if step==1:
      assert bool((ds==0).all() and (dl==0).all());assert torch.equal(shift,batch['shift']) and torch.equal(rad,batch['radius'])
     loss=-.75*(factor[:,None,None,None,None]*torch.sigmoid((rad-(remaining-shift).abs())/row['tau'])*torch.exp(-2*rad*batch['scale']/method.SIGMA_GLOBAL))[mask].mean();assert torch.isfinite(loss);loss.backward();grad=method.gradient_norm(m);assert np.isfinite(grad) and grad>0;torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optim[a].step();schedulers[a].step();emas[a].update(m);digests[a].update(ix.numpy().tobytes());row[a]=dict(loss=float(loss.detach()),gradient_norm=grad,lr=schedulers[a].get_last_lr()[0],shift_abs=float(ds.detach().abs().mean()),radius_ratio=float(dl.detach().exp().mean()))
   if step<=2 or step%200==0:history.append(row);f.write(json.dumps(row)+'\n');f.flush();budget.check();_write(output/'heartbeat.json',dict(stage='hidden_step_'+str(step),**budget.snapshot()));print(json.dumps(row),flush=True)
 for a,m in models.items():
  for endpoint,h in [('live',m),('ema_audit_only',emas[a].shadow)]:
   h.eval().requires_grad_(False)
   for p in h.parameters():p.grad=None
   assert method.all_finite(h);torch.save(dict(architecture=HiddenIntervalResidual.architecture,model_config=dict(channels=128,out_steps=20),state_dict=h.state_dict(),metadata=dict(recipe='fixed_hidden_SPS_v1',arm=a,endpoint=endpoint,steps=4000)),output/(a+'_'+endpoint+'_hidden.pt'))
 assert len({d.hexdigest() for d in digests.values()})==1
 _write(output/'training_report.json',dict(steps_per_arm=4000,initial_state=initial_hash,initial_function_bitexact=True,endpoint_states={a:method.state_hash(m) for a,m in models.items()},batch_digests={a:d.hexdigest() for a,d in digests.items()},history=history));return models


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('point','center','radius','real-dir','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--expected-cache',type=Path)
 parser.add_argument('--expected-state')
 parser.add_argument('--gpu-seconds',type=float,default=900)
 parser.add_argument('--wall-seconds',type=float,default=1200)
 args=parser.parse_args(argv)
 if min(args.gpu_seconds,args.wall_seconds)<=0:parser.error('budgets must be positive')
 args.output.mkdir(parents=True,exist_ok=False);budget=TrainingBudget(args.output,args.gpu_seconds,args.wall_seconds)
 sources={n:getattr(args,n).resolve() for n in ('point','center','radius')};pins={str(p):method.file_hash(p) for p in sources.values()};names=json.loads(args.cases.read_text())
 if not isinstance(names,list) or len(names)!=len(set(names)):raise ValueError('cases must be unique names')
 inventory={c.path.name:c for c in discover_cases(args.real_dir,exclude_files=['7575_0.h5'])}
 if set(names)-inventory.keys():raise ValueError('case not present in allowed release inventory')
 cases=[inventory[n] for n in names]
 for c in cases:pins[str(c.path.resolve())]=method.file_hash(c.path)
 _write(args.output/'job.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=list(os.sys.argv),source_pins=pins,cases=names))
 try:
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda');budget.gpu_started=time.monotonic()
  frozen={k:method.load_point(p,device) if k=='point' else load_expanded_head(p,device) if k=='radius' else method.load_head(p,device) for k,p in sources.items()};states={k:method.state_hash(m) for k,m in frozen.items()}
  expected=None if args.expected_cache is None else json.loads(args.expected_cache.read_text());cache=hidden_training_cache(frozen,method.Native79Dataset(cases),device,args.output,budget,expected);models=train_hidden_residual(frozen,cache,device,args.output,budget);state=method.state_hash(models['aligned'])
  if args.expected_state is not None and state!=args.expected_state:raise RuntimeError('fixed LIVE endpoint does not reproduce')
  assert method.state_hash(load_hidden_residual(args.output/'aligned_live_hidden.pt',device))==state
  for k,m in frozen.items():assert method.state_hash(m)==states[k] and all(p.grad is None and not p.requires_grad for p in m.parameters())
  for p,h in pins.items():assert method.file_hash(p)==h,p
  _write(args.output/'report.json',dict(status='COMPLETE',validity='VALID',endpoint_state=state,expected_state_exact=args.expected_state is not None,source_pins_unchanged=True,training_steps=4000,training_windows=len(cache['point']),evaluation_arrays_read=False,budget=budget.check()))
 except BaseException as e:
  _write(args.output/'failure.json',dict(status='DEFERRED_BY_RESOURCE' if 'RESOURCE_CAP' in str(e) else 'INVALID',exception=repr(e),budget=budget.snapshot()));raise
