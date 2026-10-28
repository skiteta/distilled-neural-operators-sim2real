"""Function-preserving additive capacity in hidden interval raw logits."""
import copy
import torch
from torch import nn
from .hidden_sps_interval import HiddenIntervalResidual

class ExpandedHiddenIntervalResidual(nn.Module):
    architecture='hidden_interval_residual_additive_2x'
    def __init__(self,channels=128,out_steps=20,source=None):
        super().__init__();self.channels=channels;self.out_steps=out_steps
        if source is None:source=HiddenIntervalResidual(channels,out_steps)
        self.base=copy.deepcopy(source.network);self.extra=copy.deepcopy(source.network)
        nn.init.zeros_(self.extra[-1].weight);nn.init.zeros_(self.extra[-1].bias)
    def forward(self,hidden):
        raw=self.base(hidden)+self.extra(hidden)
        b,_,h,w=raw.shape;value=raw.reshape(b,self.out_steps,2,2,h,w)
        return .25*torch.tanh(value[:,:,0].permute(0,1,3,4,2)),.50*torch.tanh(value[:,:,1].permute(0,1,3,4,2))

def load_hidden_capacity(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False)
    if c['architecture']==HiddenIntervalResidual.architecture:m=HiddenIntervalResidual(**c['model_config'])
    elif c['architecture']==ExpandedHiddenIntervalResidual.architecture:m=ExpandedHiddenIntervalResidual(**c['model_config'])
    else:raise ValueError('unsupported hidden architecture')
    m.load_state_dict(c['state_dict'],strict=True);return m.to(device).eval().requires_grad_(False)
