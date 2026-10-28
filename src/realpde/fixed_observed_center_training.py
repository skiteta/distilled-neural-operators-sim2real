"""Fixed 2,000-step observed center-only SPS recipe. Training inputs only.

Width, point and source heads stay frozen. No evaluation or experiment imports.
"""
from __future__ import annotations
import argparse,copy,datetime,gc,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from . import identity_interval_hardening as method
from .data import discover_cases
from .temporal_start_dataset import ReleasedTemporalStartDataset
from .fixed_sps_head_training import TrainingBudget
from .fixed_hidden_sps_training import hidden_training_cache
from .hidden_sps_interval import HiddenIntervalResidual,load_hidden_residual
from .sps_radius_capacity import load_expanded_head
ARMS=('aligned_observed',)
sha=method.file_hash
def write(output,name,value):
 p=output/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(q,p)
class Budget(TrainingBudget):
 def check(self,stage=None):
  v=self.snapshot()
  for key,limit in [('wall_seconds',self.wall_limit),('gpu_reservation_seconds',self.gpu_limit),('peak_host_bytes',20*1024**3),('peak_device_bytes',8*1024**3)]:
   if v[key]>limit:raise RuntimeError('RESOURCE_CAP '+key)
  if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())>64*1024**2:raise RuntimeError('RESOURCE_CAP output')
  if stage:write(self.output,'heartbeat.json',dict(pid=os.getpid(),stage=stage,**v))
  return v
def fit(frozen,device,real_dir,names,output,budget,expected):
 OUT=output
 train=[c for c in discover_cases(real_dir,exclude_files=['7575_0.h5']) if c.path.name in names]
 data=ReleasedTemporalStartDataset(train,stride=20,budget=budget)
 cache=hidden_training_cache(frozen,data,device,OUT,budget,expected)
 raw=torch.empty((len(data),40,32,64),dtype=torch.float32)
 for start in range(0,len(data),8):
  stop=min(start+8,len(data));x=torch.stack([data[i][0][...,:2] for i in range(start,stop)])
  z=(x-x.mean(dim=1,keepdim=True))/cache['scale'][start:stop]
  raw[start:stop]=z.permute(0,1,4,2,3).reshape(-1,40,32,64)
  budget.check()
 write(OUT,'observed_feature_cache.json',dict(rows=len(data),shape=list(raw.shape),sha256=method.tensor_hash(raw),query_only=True,time_centered=True,scale_from_observations=True))
 del data;gc.collect()
 # Convert base diagnostics to the accepted frozen hidden intervals exactly once.
 for start in range(0,len(raw),8):
  stop=min(start+8,len(raw))
  with torch.inference_mode():
   ds,dl=frozen['hidden'](cache['hidden'][start:stop].to(device))
   cache['shift'][start:stop].add_(ds.cpu());cache['radius'][start:stop].mul_(dl.exp().cpu())
  budget.check()
 cache['observed_raw']=raw
 with torch.random.fork_rng(devices=[]):torch.random.default_generator.manual_seed(20260908608);initial=HiddenIntervalResidual(channels=168)
 models={a:copy.deepcopy(initial).to(device) for a in ARMS}
 opts={a:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for a,m in models.items()};schedulers={a:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=2000,eta_min=2e-6) for a,o in opts.items()};rng=torch.Generator().manual_seed(20260908608);digest=hashlib.sha256();history=[]
 with (OUT/'training_curve.jsonl').open('x') as f:
  for step in range(1,2001):
   budget.check();ix=torch.randint(len(raw),(8,),generator=rng);digest.update(ix.numpy().tobytes());b={k:v.index_select(0,ix).to(device) for k,v in cache.items()};row=dict(step=step,tau=.04*10**(-(step-1)/1999))
   with budget.cuda():
    factor=method.sample_factor(b['point'],b['target']);remaining=(b['target']-b['point'])/b['scale'];mask=b['target']!=0
    for a,m in models.items():
     opts[a].zero_grad(set_to_none=True);feature=b['observed_raw'] if a=='aligned_observed' else torch.roll(b['observed_raw'],(16,32),(2,3));ds,dl=m(torch.cat((b['hidden'],feature),dim=1));shift=b['shift']+ds;radius=b['radius']
     if step==1:assert torch.equal(shift,b['shift']) and torch.equal(radius,b['radius'])
     loss=-.75*(factor[:,None,None,None,None]*torch.sigmoid((radius-(remaining-shift).abs())/row['tau'])*torch.exp(-2*radius*b['scale']/method.SIGMA_GLOBAL))[mask].mean();assert torch.isfinite(loss);loss.backward();grad=method.gradient_norm(m);assert np.isfinite(grad) and grad>0;torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[a].step();schedulers[a].step();row[a]=dict(loss=float(loss.detach()),gradient_norm=grad,shift_abs=float(ds.detach().abs().mean()),radius_ratio=float(dl.detach().exp().mean()))
   if step%500==0:
    checkpoint=dict(step=step,models={a:m.state_dict() for a,m in models.items()},optimizers={a:o.state_dict() for a,o in opts.items()},schedulers={a:o.state_dict() for a,o in schedulers.items()},batch_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,recipe_source_sha256=sha(Path(__file__)),budget=budget.snapshot());tmp=OUT/'checkpoint.tmp';torch.save(checkpoint,tmp);os.replace(tmp,OUT/('checkpoint_'+str(step)+'.pt'))
   if step<=2 or step%200==0:history.append(row);f.write(json.dumps(row)+'\n');f.flush();budget.check('train_'+str(step));print(json.dumps(row),flush=True)
 for a,m in models.items():
  m.eval().requires_grad_(False)
  for p in m.parameters():p.grad=None
  torch.save(dict(architecture=HiddenIntervalResidual.architecture,model_config=dict(channels=168,out_steps=20),state_dict=m.state_dict(),metadata=dict(steps=2000,endpoint='LIVE',arm=a)),OUT/(a+'_live.pt'))
 write(OUT,'training_report.json',dict(steps_per_arm=2000,train_only=True,batch_digest=digest.hexdigest(),same_initial_weights=True,initial_function_exact=True,endpoint_states={a:method.state_hash(m) for a,m in models.items()},history=history))
 del cache;gc.collect();torch.cuda.empty_cache();return models

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('point','center','radius','hidden','real-dir','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--expected-cache',type=Path);parser.add_argument('--expected-state');parser.add_argument('--gpu-seconds',type=float,default=300);parser.add_argument('--wall-seconds',type=float,default=420)
 args=parser.parse_args(argv);assert args.gpu_seconds>0 and args.wall_seconds>0;args.output.mkdir(parents=True,exist_ok=False);budget=Budget(args.output,args.gpu_seconds,args.wall_seconds)
 names=json.loads(args.cases.read_text());assert isinstance(names,list) and len(names)==len(set(names));available={c.path.name:c for c in discover_cases(args.real_dir,exclude_files=['7575_0.h5'])};assert not set(names)-set(available)
 sources={k:getattr(args,k) for k in ('point','center','radius','hidden')};pins={str(p):sha(p) for p in [*sources.values(),args.cases,Path(__file__),*[available[n].path for n in names]]};write(args.output,'job.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arguments=vars(args)|{k:str(v) for k,v in vars(args).items() if isinstance(v,Path)},sources=pins,case_order=[c.path.name for c in discover_cases(args.real_dir,exclude_files=['7575_0.h5']) if c.path.name in names]))
 def timeout(*_):raise TimeoutError('RESOURCE_CAP wall')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(int(args.wall_seconds))
 try:
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda');budget.gpu_started=time.monotonic()
  frozen={k:method.load_point(p,device) if k=='point' else method.load_head(p,device) if k=='center' else load_expanded_head(p,device) if k=='radius' else load_hidden_residual(p,device) for k,p in sources.items()};states={k:method.state_hash(m) for k,m in frozen.items()}
  expected=None if args.expected_cache is None else json.loads(args.expected_cache.read_text());models=fit(frozen,device,args.real_dir,names,args.output,budget,expected);state=method.state_hash(models['aligned_observed']);assert args.expected_state is None or state==args.expected_state,'endpoint differs'
  for k,m in frozen.items():assert method.state_hash(m)==states[k] and all(p.grad is None and not p.requires_grad for p in m.parameters())
  for p,h in pins.items():assert sha(p)==h,p
  assert method.state_hash(load_hidden_residual(args.output/'aligned_observed_live.pt',device))==state
  write(args.output,'report.json',dict(status='COMPLETE',validity='VALID',endpoint_state=state,expected_state_exact=args.expected_state is not None,steps=2000,source_pins_unchanged=True,evaluation_arrays_read=False,budget=budget.check('complete')))
 except BaseException as e:write(args.output,'failure.json',dict(status='DEFERRED_BY_RESOURCE' if 'RESOURCE_CAP' in str(e) else 'INVALID',exception=repr(e),budget=budget.snapshot()));raise
 finally:signal.alarm(0);write(args.output,'attempt_budget.json',budget.snapshot())
if __name__=='__main__':main()
