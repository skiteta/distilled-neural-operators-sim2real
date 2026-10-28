"""Small baseline-specific amplitude correction, with exact compact error algebra."""
import math
import numpy as np
import torch
from torch import nn

FEATURE_NAMES=('baseline_log_energy','observed_log_ratio','early_log_ratio','late_log_ratio',
               'half_disagreement','baseline_mean_u','baseline_mean_v','observed_mean_u','observed_mean_v','observed_mask')
OBSERVED_COLUMNS=(1,2,3,4,7,8,9)
LOG_CAP=math.log(1.1)

def energy(x):
    x=np.asarray(x,dtype=np.float64)
    return .5*np.mean(np.sum((x-x.mean(0))**2,axis=-1),axis=0)

def features(observed,baseline):
    observed=np.asarray(observed);baseline=np.asarray(baseline)
    if observed.shape!=baseline.shape or observed.ndim!=5 or observed.shape[1:]!=(20,32,64,2):raise ValueError('Matching B,20,32,64,2 arrays required')
    out=[]
    for x,b in zip(observed,baseline):
        x=x.astype(np.float64);b=b.astype(np.float64);kb=energy(b);ko=energy(x);ke=energy(x[:10]);kl=energy(x[10:]);unit=max(float(kb.mean()),1e-12);eps=1e-6*unit+1e-16;rms=max(float(np.sqrt(np.mean(x*x))),1e-8);bm=b.mean(0);xm=x.mean(0)
        f=np.stack([np.log((kb+eps)/(unit+eps)),np.log((ko+eps)/(kb+eps)),np.log((ke+eps)/(kb+eps)),np.log((kl+eps)/(kb+eps)),np.abs(ke-kl)/(ko+eps),bm[...,0]/rms,bm[...,1]/rms,xm[...,0]/rms,xm[...,1]/rms,np.any(x!=0,axis=(0,3)).astype(float)],axis=-1)
        out.append(f)
    out=np.array(out)
    if not np.isfinite(out).all():raise ValueError('Nonfinite observable features')
    return out

def compact_statistics(baseline,target):
    b=np.asarray(baseline,dtype=np.float64);y=np.asarray(target,dtype=np.float64);f=b-b.mean(1,keepdims=True);e=b-y
    return dict(A=np.sum(f*f,axis=(1,4)),D=np.sum(f*e,axis=(1,4)),E=np.sum(e*e,axis=(1,2,3,4)),target_norm=np.maximum(np.sqrt(np.sum(y*y,axis=(1,2,3,4))),1e-8),Kb=np.array([energy(z) for z in b]),Ky=np.array([energy(z) for z in y]),target_K_norm=np.maximum(np.sqrt(np.sum(np.array([energy(z) for z in y])**2,axis=(1,2))),1e-8))

def error_factors(scale,stats):
    delta=scale-1
    sq=stats['E']+(2*delta*stats['D']+delta.square()*stats['A']).sum((1,2))
    rel=sq.clamp_min(1e-24).sqrt()/stats['target_norm']
    tke=(scale.square()*stats['Kb']-stats['Ky']).square().sum((1,2)).clamp_min(1e-24).sqrt()/stats['target_K_norm']
    # Official SPS accuracy factor normalizes error by 0.5, unlike score_error().
    return rel,tke,.5/(1+2*rel)+.3/(1+2*tke)

class AmplitudeHead(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.register_buffer('feature_mean',torch.as_tensor(mean,dtype=torch.float64));self.register_buffer('feature_scale',torch.as_tensor(scale,dtype=torch.float64));self.first=nn.Linear(10,16,dtype=torch.float64);self.last=nn.Linear(16,1,dtype=torch.float64);nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,x):return torch.exp(LOG_CAP*torch.tanh(self.last(torch.tanh(self.first((x-self.feature_mean)/self.feature_scale)))[...,0]))
    def export(self):return {k:v.detach().cpu().numpy().tolist() for k,v in self.state_dict().items()}

def numpy_scales(model,x,rolled=False):
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in model.items()};values=[]
    for row in np.asarray(x,dtype=np.float64):
        row=row.copy()
        if rolled:row[...,list(OBSERVED_COLUMNS)]=np.roll(row[...,list(OBSERVED_COLUMNS)],(16,32),(0,1))
        z=(row-arrays['feature_mean'])/arrays['feature_scale'];h=np.tanh(z@arrays['first.weight'].T+arrays['first.bias']);raw=(h@arrays['last.weight'].T+arrays['last.bias'])[...,0];values.append(np.exp(LOG_CAP*np.tanh(raw)))
    return np.array(values)

def apply_scales(baseline,scale):
    b=np.asarray(baseline,dtype=np.float32);s=np.asarray(scale,dtype=np.float64)
    if s.shape!=b.shape[:1]+b.shape[2:4] or not np.isfinite(s).all() or (s<=0).any():raise ValueError('Finite positive percell scales required')
    result=[]
    for x,z in zip(b,s):
        if np.all(z==1):result.append(x.copy());continue
        d=x.astype(np.float64);mu=d.mean(0,keepdims=True);result.append((mu+(d-mu)*z[None,:,:,None]).astype(np.float32))
    return np.array(result)
