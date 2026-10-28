"""Online frozen features for temporal-start-density-only hidden SPS training."""
import copy,json
import numpy as np
import torch
from . import identity_interval_hardening as method
from .hidden_sps_interval import HiddenIntervalResidual,predict_with_hidden
from .fixed_hidden_sps_training import train_hidden_residual
from .fixed_sps_head_training import _write

KEYS=('point','target','hidden','scale','shift','radius')
def features(frozen,items,device):
    x=torch.stack([i[0] for i in items]).to(device)
    target=torch.stack([i[1][...,:2] for i in items]).to(device)
    with torch.no_grad():
        p,h=predict_with_hidden(frozen['point'],x)
        d,scale=method.flow_diagnostics(x,p[...,:2])
        shift,_=frozen['center'](d);_,radius=frozen['radius'](d)
        values=dict(point=p[...,:2],target=target,hidden=h,scale=scale,shift=shift,radius=radius)
        return {k:v.detach().contiguous().clone() for k,v in values.items()}

class OnlineColumns:
    def __init__(self,frozen,dataset,device):
        self.frozen=frozen;self.dataset=dataset;self.device=device;self.indices=None;self.batch=None
    class Column:
        def __init__(self,parent,key):self.parent=parent;self.key=key
        def __len__(self):return len(self.parent.dataset)
        def index_select(self,dim,indices):
            assert dim==0
            p=self.parent
            if p.indices is None or not torch.equal(indices,p.indices):
                p.batch=features(p.frozen,[p.dataset[int(i)] for i in indices],p.device);p.indices=indices.clone()
            return p.batch[self.key]
    def __getitem__(self,key):return self.Column(self,key)
    def items(self):return [(k,self[k]) for k in KEYS]

def native_probe(frozen,native,dense,device,output,expected):
    indices=torch.randint(len(native),(8,),generator=torch.Generator().manual_seed(20260907586))
    lookup={w:i for i,w in enumerate(dense.windows)}
    online=features(frozen,[dense[lookup[native.windows[int(i)]]] for i in indices],device)
    # Reconstruct the original ordered B8 cache blocks that supplied this random batch.
    blocks={}
    for i in indices.tolist():
        start=i//8*8
        if start not in blocks:blocks[start]=features(frozen,[native[j] for j in range(start,min(start+8,len(native)))],device)
    cache={k:torch.stack([blocks[i//8*8][k][i%8] for i in indices.tolist()]) for k in KEYS}
    exact={k:torch.equal(cache[k],online[k]) for k in KEYS}
    errors={k:float((cache[k]-online[k]).abs().max()) for k in KEYS}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026081702);model=HiddenIntervalResidual(128,20).to(device)
    factor=method.sample_factor(online['point'],online['target']);remaining=(online['target']-online['point'])/online['scale'];mask=online['target']!=0
    ds,dl=model(online['hidden']);ss=online['shift']+ds;rr=online['radius']*dl.exp()
    loss=-.75*(factor[:,None,None,None,None]*torch.sigmoid((rr-(remaining-ss).abs())/.04)*torch.exp(-2*rr*online['scale']/method.SIGMA_GLOBAL))[mask].mean();loss.backward();grad=method.gradient_norm(model)
    report=dict(indices=indices.tolist(),features_bitexact=exact,max_abs=errors,loss=float(loss.detach()),gradient_norm=grad,expected=expected,old_first_loss_exact=float(loss.detach())==expected['loss'],old_first_gradient_exact=grad==expected['gradient_norm'])
    _write(output/'native_online_probe.json',report)
    assert all(exact.values()) and report['old_first_loss_exact'] and report['old_first_gradient_exact'],'native online path does not reproduce existing first step'
    return report

def train_dense_hidden(frozen,dataset,device,output,budget):
    return train_hidden_residual(frozen,OnlineColumns(frozen,dataset,device),device,output,budget)
