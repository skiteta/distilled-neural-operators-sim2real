"""Restore baseline temporal mean and optionally local TKE without labels."""
import numpy as np


def restore_baseline_moments(candidate,baseline,preserve_tke=True):
    candidate=np.asarray(candidate,dtype=np.float32);baseline=np.asarray(baseline,dtype=np.float32)
    if candidate.shape!=baseline.shape or candidate.ndim!=5 or candidate.shape[1]!=20 or candidate.shape[-1]!=2:
        raise ValueError('Expected matching B,20,H,W,2 trajectories')
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError('Nonfinite trajectories')
    result=np.empty_like(candidate)
    for i,(a,b) in enumerate(zip(candidate,baseline)):
        if np.array_equal(a,b):result[i]=b;continue
        a=a.astype(np.float64);b=b.astype(np.float64);mu=b.mean(0,keepdims=True);fb=b-mu;fa=a-a.mean(0,keepdims=True)
        if preserve_tke:
            ea=np.mean(np.sum(fa*fa,axis=-1),axis=0);eb=np.mean(np.sum(fb*fb,axis=-1),axis=0)
            scale=np.sqrt(np.divide(eb,ea,out=np.zeros_like(eb),where=ea>0))
            fa=np.where((ea>0)[None,:,:,None],fa*scale[None,:,:,None],fb)
        result[i]=(mu+fa).astype(np.float32)
    return result
