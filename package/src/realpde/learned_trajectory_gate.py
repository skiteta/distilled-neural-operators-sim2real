"""Window-local mixture gate for frozen baseline, Q1 mean and Q2 mean experts."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn

FEATURE_NAMES = ('log_rms','mean_u_scaled','mean_v_scaled','fluct_u_scaled','fluct_v_scaled',
                 'time_corr1','time_corr5','time_increment_scaled','x_increment_scaled','y_increment_scaled',
                 'late_early_u_scaled','late_early_v_scaled','wake_fluctuation_logratio','observed_valid_fraction',
                 'Q1_baseline_TKE_logratio','Q2_baseline_TKE_logratio')

def mixture_grid():
    triples=[(i/4,j/4,(4-i-j)/4) for i in range(5) for j in range(5-i)]
    return np.array([(1.,0.,0.)]+[v for v in triples if v!=(1.,0.,0.)],dtype=np.float32)

def case_folds(names,folds=5,seed=6381):
    ordered=sorted(names,key=lambda n:hashlib.sha256((str(seed)+n).encode()).hexdigest())
    return {name:i%folds for i,name in enumerate(ordered)}

def case_weights(rows):
    names=[r['case'] for r in rows];counts={n:names.count(n) for n in set(names)}
    return np.array([1/(len(counts)*counts[n]) for n in names],dtype=np.float64)

def window_features(observed,baseline,q1,q2):
    arrays=[np.asarray(v) for v in (observed,baseline,q1,q2)]
    if any(a.shape!=arrays[0].shape for a in arrays) or arrays[0].shape[1:]!=(20,32,64,2):
        raise ValueError('Expected matching B,20,32,64,2 arrays')
    output=[];eps=1e-16
    def rms(v):return float(np.sqrt(np.mean(v*v)+eps))
    def tke(v):return float(np.mean((v-v.mean(0,keepdims=True))**2))
    def corr(a,b):return float(np.sum(a*b)/np.sqrt(max(float(np.sum(a*a)*np.sum(b*b)),eps)))
    for values in zip(*arrays):
        x,b,a,c=[np.asarray(v,dtype=np.float64) for v in values];scale=rms(x);f=x-x.mean(0,keepdims=True);mean=x.mean((0,1,2));drift=x[10:].mean((0,1,2))-x[:10].mean((0,1,2));bt=tke(b)
        output.append([np.log(scale),mean[0]/scale,mean[1]/scale,rms(f[...,0])/scale,rms(f[...,1])/scale,
                       corr(f[1:],f[:-1]),corr(f[5:],f[:-5]),rms(x[1:]-x[:-1])/scale,
                       rms(x[:,:,1:]-x[:,:,:-1])/scale,rms(x[:,1:]-x[:,:-1])/scale,
                       drift[0]/scale,drift[1]/scale,np.log((np.mean(f[:,:,32:]**2)+eps)/(np.mean(f[:,:,:32]**2)+eps)),
                       np.mean(np.any(x!=0,axis=(0,3))),np.log((tke(a)+eps)/(bt+eps)),np.log((tke(c)+eps)/(bt+eps))])
    result=np.asarray(output,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite gate features')
    return result

class GainGate(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(16,16),nn.Tanh(),nn.Linear(16,14))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('feature_mean',torch.zeros(16));self.register_buffer('feature_scale',torch.ones(16));self.register_buffer('gain_scale',torch.ones(()));self.double()
    def forward(self,x):
        values=self.network((x-self.feature_mean)/self.feature_scale)*self.gain_scale
        return torch.cat((torch.zeros((len(x),1),dtype=values.dtype,device=values.device),values),1)
    @torch.inference_mode()
    def predict_gain(self,features):
        # Fixed one-row arithmetic makes choices independent of caller batching.
        return np.concatenate([self(torch.as_tensor(row[None],dtype=torch.float64)).numpy() for row in np.asarray(features)],axis=0)

def decisions(gains,allowed=None):
    indices=np.arange(15) if allowed is None else np.asarray(allowed)
    chosen=indices[np.argmax(np.asarray(gains)[:,indices],axis=1)]
    return chosen,mixture_grid()[chosen]

def combine(experts,weights):
    experts=np.asarray(experts,dtype=np.float32);weights=np.asarray(weights,dtype=np.float32)
    if experts.shape[1]!=3 or weights.shape!=(len(experts),3):raise ValueError('Expected3experts and B,3 weights')
    if not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1,rtol=0,atol=1e-7):raise ValueError('Invalid convex weights')
    result=np.zeros_like(experts[:,0])
    for k in range(3):result+=experts[:,k]*weights[:,k,None,None,None,None]
    return result
