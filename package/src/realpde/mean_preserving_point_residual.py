"""One-pass hidden point correction, optionally preserving temporal mean.

The existing absolute interval path stays unchanged. This head changes only
submitted u/v, using the same frozen point forward and call-scoped hidden field.
"""
import torch
from torch import nn

class HiddenPointResidual(nn.Module):
    architecture='hidden_point_residual_v1'
    def __init__(self,channels=128,out_steps=20,preserve_mean=True):
        super().__init__();self.channels=channels;self.out_steps=out_steps;self.preserve_mean=preserve_mean
        self.network=nn.Sequential(nn.Conv2d(channels,channels,1),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1,groups=channels),nn.GELU(),nn.Conv2d(channels,out_steps*2,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,hidden,scale):
        raw=self.network(hidden);n,_,h,w=raw.shape
        delta=.25*torch.tanh(raw.reshape(n,self.out_steps,2,h,w).permute(0,1,3,4,2))*scale
        if self.preserve_mean:delta=delta-delta.mean(dim=1,keepdim=True)
        return delta

def load_point_residual(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['architecture']!=HiddenPointResidual.architecture:raise ValueError('architecture mismatch')
    model=HiddenPointResidual(**value['model_config']);model.load_state_dict(value['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False)
