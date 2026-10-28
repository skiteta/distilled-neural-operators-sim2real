"""Released simulation replay with a focus stream at a fixed per-window emphasis (training.sim_focus_ratio r): the focus fraction is f = r Wf/(r Wf + Wo) and each batch of 8 takes on average 8f windows (accumulator)
from the simulated regimes whose real counterpart is NOT in this run's real training set (held real cases and the regimes
without released real data) and the rest from the other simulations, each from its own shuffled order. Same checks,
provenance and loss as sim_replay_all.py (point RelL2 on simulated futures, weight sim_weight)."""
from pathlib import Path
import hashlib
import torch
from realpde.data import discover_cases, FlowWindowDataset
from realpde.temporal_start_dataset import ReleasedTemporalStartDataset
from realpde.metrics import relative_l2
from experiment_records import write, sha, rel


class SimulationReplay:
    def __init__(self, root, manifest, destination, budget, device, policy='train_only'):
        self.device=device;self.destination=destination;self.seed=manifest['training']['seed']
        self.weight=float(manifest['training'].get('sim_weight',.1));self.calls=0;self.samples=0;self.cursor=0
        self.digest=hashlib.sha256();self.seen=set();self.first=None;self.last_loss=None
        allowed=set(manifest['train_cases']);excluded=set(manifest['eval_cases'])
        every=discover_cases(root/'data/extracted/train_sim',exclude_files=['7575_0.h5'])
        if policy=='train_only':cases=[c for c in every if c.path.name in allowed];assert {c.path.name for c in cases}==allowed
        elif policy=='all':cases=list(every)   # Sim2Real: every released simulation, including regimes without released real data and the held regimes
        elif policy=='exclude_eval':cases=[c for c in every if c.path.name not in excluded]
        else:raise ValueError(policy)
        self.policy=policy
        # Real and simulated files share the Re_AoA naming; overlap with held real cases is by regime only (different data).
        self.data=ReleasedTemporalStartDataset(cases,stride=1,budget=budget)
        self.data.windows=[(i,s) for i,c in enumerate(cases) for s in range(c.frames-40+1)]
        checks=[]
        for ci in [0,len(cases)-1]:
            native=FlowWindowDataset([cases[ci]],stride=1)
            for start in [0,(cases[ci].frames-40)//2,cases[ci].frames-40]:
                pos=self.data.windows.index((ci,start));a,b,_=self.data[pos];x,y=native[start]
                assert torch.equal(a,x) and torch.equal(b,y)
                checks.append(dict(case=cases[ci].path.name,start=start,first_target=start+20,last_target=start+39))
        self.rng=torch.Generator().manual_seed(self.seed+19)
        self.focus_ratio=float(manifest['training']['sim_focus_ratio']);assert self.focus_ratio>=1.0   # per-window weight of focus vs other sims
        self.focus_names=sorted({c.path.name for c in cases}-allowed)   # sims whose real counterpart is not trained on
        fset={i for i,c in enumerate(cases) if c.path.name in set(self.focus_names)}
        self.focus_pool=torch.tensor([p for p,(i,s) in enumerate(self.data.windows) if i in fset],dtype=torch.long)
        self.other_pool=torch.tensor([p for p,(i,s) in enumerate(self.data.windows) if i not in fset],dtype=torch.long)
        assert len(self.focus_pool)>0 and len(self.other_pool)>0
        wf,wo=len(self.focus_pool),len(self.other_pool);self.focus_fraction=self.focus_ratio*wf/(self.focus_ratio*wf+wo)
        self.acc=0.0;self.k=None   # per-batch focus count follows the exact fraction on average (accumulator)
        self.frng=torch.Generator().manual_seed(self.seed+29)
        self.forder=self.focus_pool[torch.randperm(len(self.focus_pool),generator=self.frng)];self.fcursor=0
        self.order=self.other_pool[torch.randperm(len(self.other_pool),generator=self.rng)]
        with torch.random.fork_rng(devices=[device]):
            torch.cuda.manual_seed(self.seed+23);self.cuda_rng=torch.cuda.get_rng_state(device)
        self.provenance=dict(source='Existing released train_sim only; no generated trajectories or external data.',policy=policy,
                             train_cases=[c.path.name for c in cases],held_real_cases=sorted(excluded),
                             simulated_regimes_matching_held_real=sorted({c.path.name for c in cases}&excluded),
                             simulated_regimes_without_released_real=sorted({c.path.name for c in cases}-allowed-excluded),teacher=None,sim_target_used_only_for_training=True,
                             source_hashes={rel(c.path):sha(c.path) for c in cases},windows=len(self.data),
                             input_frames=20,target_frames=20,native_windows_checked=checks,
                             pressure='Input pressure set to zero to match released real inputs; velocity targets unchanged.',
                             sampling='Independent shuffled simulation window stream; real sampler/reflection/dropout RNG stream is preserved.',
                             weight=self.weight,storage_bytes=self.data.storage_bytes,
                             focus=dict(ratio=self.focus_ratio,fraction=self.focus_fraction,per_batch='accumulator',cases=self.focus_names,focus_windows=len(self.focus_pool),other_windows=len(self.other_pool)))
        write(destination/'simulation_replay_source.json',self.provenance)

    def loss(self,point,heads):
        prev=self.acc;self.acc+=8*self.focus_fraction;k=int(self.acc)-int(prev);k=max(0,min(8,k))
        if self.cursor+(8-k)>len(self.order):
            self.order=self.other_pool[torch.randperm(len(self.other_pool),generator=self.rng)];self.cursor=0
        if self.fcursor+k>len(self.forder):
            self.forder=self.focus_pool[torch.randperm(len(self.focus_pool),generator=self.frng)];self.fcursor=0
        indices=torch.cat((self.forder[self.fcursor:self.fcursor+k],self.order[self.cursor:self.cursor+8-k]))
        self.fcursor+=k;self.cursor+=8-k
        self.seen.update(indices.tolist());self.digest.update(indices.numpy().tobytes())
        items=[self.data[i] for i in indices.tolist()]
        x=torch.stack([v[0] for v in items]).to(self.device);target=torch.stack([v[1] for v in items]).to(self.device)
        x[...,2]=0
        real_state=torch.cuda.get_rng_state(self.device)
        with torch.random.fork_rng(devices=[self.device]):
            torch.cuda.set_rng_state(self.cuda_rng,self.device)
            prediction=point(x);loss=relative_l2(prediction,target)
            self.cuda_rng=torch.cuda.get_rng_state(self.device)
        assert torch.equal(real_state,torch.cuda.get_rng_state(self.device))
        if self.calls==0:
            parameters=tuple(point.parameters());g=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            norm=sum(float(v.norm()) for v in g if v is not None);assert norm>0
            hg=torch.autograd.grad(loss,tuple(heads.parameters()),retain_graph=True,allow_unused=True)
            assert all(v is None for v in hg)
            self.first=dict(point_gradient_norm=norm,all_interval_gradients_none=True,
                            real_CUDA_RNG_unchanged=True,pressure_zero=True,
                            first_sample_ids=[[self.data.cases[self.data.windows[i][0]].path.name,self.data.windows[i][1]] for i in indices.tolist()],
                            auxiliary_loss=float(loss.detach()),weighted_loss=float(loss.detach())*self.weight)
            write(self.destination/'simulation_replay_first_checks.json',self.first)
        assert torch.isfinite(loss)
        self.calls+=1;self.samples+=len(indices);self.last_loss=float(loss.detach())
        return self.weight*loss

    def state(self):
        return dict(calls=self.calls,samples=self.samples,unique_windows=len(self.seen),digest=self.digest.hexdigest(),
                    rng=self.rng.get_state(),order=self.order,cursor=self.cursor,cuda_rng=self.cuda_rng)

    def report(self):
        return dict(policy=self.policy,calls=self.calls,samples=self.samples,unique_windows=len(self.seen),windows=len(self.data),
                    digest=self.digest.hexdigest(),weight=self.weight,last_loss=self.last_loss,
                    source=rel(self.destination/'simulation_replay_source.json'),first_checks=self.first)
