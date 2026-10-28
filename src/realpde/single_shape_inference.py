"""Single frozen head already projects mean/TKE; avoid a second CPU projection."""
import numpy as np
import torch

@torch.inference_mode()
def predict_single_shape(head, observed, baseline, microbatch=16):
    if len(observed)!=len(baseline):raise ValueError('length mismatch')
    if not len(baseline):return np.empty_like(baseline)
    device=next(head.parameters()).device;parts=[]
    for start in range(0,len(baseline),microbatch):
        n=min(microbatch,len(baseline)-start)
        xx=np.zeros((microbatch,*observed.shape[1:]),dtype=np.float32);bb=np.zeros((microbatch,*baseline.shape[1:]),dtype=np.float32)
        xx[:n]=observed[start:start+n];bb[:n]=baseline[start:start+n]
        value=head(torch.from_numpy(xx).to(device),torch.from_numpy(bb).to(device))
        parts.append(value[:n].cpu().numpy())
    return np.concatenate(parts)
