"""Baseline feature reuse for a bounded mean-preserving amplitude head."""
import math
import numpy as np
import torch
from torch import nn
from realpde.baseline_matched_tke import LOG_CAP

class HiddenAmplitudeHead(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.register_buffer('feature_mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('feature_scale',torch.as_tensor(scale,dtype=torch.float32));self.first=nn.Linear(178,16);self.last=nn.Linear(16,1);nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,x):
        z=self.last(torch.tanh(self.first((x-self.feature_mean)/self.feature_scale)))[...,0]
        return torch.exp(LOG_CAP*torch.tanh(z))
    def export(self):return {k:v.detach().cpu().numpy().tolist() for k,v in self.state_dict().items()}

def numpy_scales(model,features,rolled=False):
    d={k:np.asarray(v,dtype=np.float64) for k,v in model.items()};out=[]
    for x in features:
        x=np.asarray(x,dtype=np.float64).copy()
        if rolled:x[...,10:]=np.roll(x[...,10:],(16,32),(0,1))
        h=np.tanh(((x-d['feature_mean'])/d['feature_scale'])@d['first.weight'].T+d['first.bias']);out.append(np.exp(LOG_CAP*np.tanh((h@d['last.weight'].T+d['last.bias'])[...,0])))
    return np.array(out)
