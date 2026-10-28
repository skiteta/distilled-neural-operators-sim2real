"""Reuse an expert's learned conditioning while transferring its residual."""
import numpy as np


def transfer_residual(native_expert, native_baseline, destination_baseline):
    q,b,d=(np.asarray(v,dtype=np.float32) for v in (native_expert,native_baseline,destination_baseline))
    if q.shape!=b.shape or q.shape!=d.shape or q.ndim!=5 or q.shape[1:]!=(20,32,64,2):
        raise ValueError('Matching B,20,32,64,2 trajectories required')
    if not all(np.isfinite(v).all() for v in (q,b,d)):raise ValueError('Nonfinite trajectories')
    return d+(q-b)
