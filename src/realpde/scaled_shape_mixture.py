"""Input-only, multiscale shape correction around a protected C659 anchor."""
import numpy as np
import torch
from torch import nn
from .baseline_shape_correction import ShapeHead, preserve_moments

class FieldExpert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(124,width,3,padding=1),nn.GELU(),
            nn.Conv2d(width,width,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(width,width,3,padding=4,dilation=4),nn.GELU(),nn.Conv2d(width,40,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)

class ScaledShapeMixture(nn.Module):
    def __init__(self,kind,anchor_state):
        super().__init__(); assert kind in ('shared','moe'); self.kind=kind
        self.anchor=ShapeHead(); self.anchor.load_state_dict(anchor_state); self.anchor.requires_grad_(False)
        self.experts=nn.ModuleList([FieldExpert(64)] if kind=='shared' else [FieldExpert(40),FieldExpert(40)])
        self.gate=nn.Sequential(nn.Linear(248,16),nn.GELU(),nn.Linear(16,2)) if kind=='moe' else None
        if self.gate is not None:
            nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
    @staticmethod
    def inputs(observed,baseline):
        fo=observed-observed.mean(1,keepdim=True); fb=baseline-baseline.mean(1,keepdim=True)
        norm=(.5*(fo.square().mean((1,2,3,4),keepdim=True)+fb.square().mean((1,2,3,4),keepdim=True))).sqrt().clamp_min(1e-6)
        pack=lambda z:z.permute(0,1,4,2,3).flatten(1,2)
        h,w=baseline.shape[2:4]; yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=baseline.device),torch.linspace(-1,1,w,device=baseline.device),indexing='ij')
        coords=torch.stack([yy,xx])[None].expand(len(baseline),-1,-1,-1)
        features=torch.cat([pack(observed/norm),pack(fo/norm),pack(fb/norm),pack(baseline.mean(1,keepdim=True)/norm),coords],1)
        return features,fb.square().sum(-1,keepdim=True).mean(1,keepdim=True).sqrt()
    def forward(self,observed,baseline,anchor=None,return_gate=False):
        if anchor is None:
            with torch.no_grad(): anchor=self.anchor(observed,baseline)
        features,rms=self.inputs(observed,baseline)
        if self.gate is None:
            z=self.experts[0](features); weights=torch.ones((len(features),1),device=features.device)
        else:
            pooled=torch.cat([features.mean((2,3)),features.square().mean((2,3)).clamp_min(1e-12).sqrt()],1)
            weights=torch.softmax(self.gate(pooled),1)
            z=sum(expert(features)*weights[:,i,None,None,None] for i,expert in enumerate(self.experts))
        z=z.reshape(len(z),20,2,z.shape[2],z.shape[3]).permute(0,1,3,4,2)
        prediction=preserve_moments(anchor+.2*z.tanh()*rms,baseline)
        return (prediction,weights) if return_gate else prediction
    @torch.inference_mode()
    def predict_numpy(self,observed,baseline,metadata=None):
        if len(observed)!=len(baseline): raise ValueError('length mismatch')
        if not len(observed): return np.empty_like(baseline)
        device=next(self.parameters()).device
        return np.concatenate([self(torch.as_tensor(np.array(x[None],copy=True),device=device),torch.as_tensor(np.array(b[None],copy=True),device=device)).cpu().numpy() for x,b in zip(observed,baseline)])
