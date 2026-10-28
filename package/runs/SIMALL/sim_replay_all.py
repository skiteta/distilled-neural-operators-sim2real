"""Released simulation replay within the current nominal-case training boundary."""
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
        self.order=torch.randperm(len(self.data),generator=self.rng)
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
                             weight=self.weight,storage_bytes=self.data.storage_bytes)
        write(destination/'simulation_replay_source.json',self.provenance)

    def loss(self,point,heads):
        if self.cursor>=len(self.data):
            self.order=torch.randperm(len(self.data),generator=self.rng);self.cursor=0
        indices=self.order[self.cursor:self.cursor+8];self.cursor+=len(indices)
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
