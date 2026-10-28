"""Fixed padded microbatches and shared anchor; no learned or target-based changes."""
import numpy as np
import torch
from .trajectory_moment_projection import restore_baseline_moments

@torch.inference_mode()
def predict_shape_ensemble(heads, observed, baseline, microbatch=16):
    if not heads or len(observed)!=len(baseline):raise ValueError('invalid ensemble inputs')
    if not len(baseline):return np.empty_like(baseline)
    device=next(heads[0].parameters()).device;parts=[]
    for start in range(0,len(baseline),microbatch):
        n=min(microbatch,len(baseline)-start)
        xx=np.zeros((microbatch,*observed.shape[1:]),dtype=np.float32);bb=np.zeros((microbatch,*baseline.shape[1:]),dtype=np.float32)
        xx[:n]=observed[start:start+n];bb[:n]=baseline[start:start+n]
        x=torch.from_numpy(xx).to(device);b=torch.from_numpy(bb).to(device)
        anchor=heads[0].anchor(x,b)
        mean=sum(head(x,b,anchor).double() for head in heads)/len(heads)
        parts.append(mean[:n].cpu().numpy())
    return restore_baseline_moments(np.concatenate(parts),baseline)
