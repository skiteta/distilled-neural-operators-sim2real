"""Observable action-size normalization for case-calibrated TKE optimism."""
import numpy as np
from .crossfit_tke_calibration import case_optimism_margin


def tke_action_scale(candidate,baseline):
    a=np.asarray(candidate);b=np.asarray(baseline)
    if a.shape!=b.shape or a.ndim!=5 or a.shape[1]!=20 or a.shape[-1]!=2:raise ValueError('Expected matching B,20,H,W,2 arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite trajectories')
    out=[]
    for x,y in zip(a,b):
        x=x.astype(np.float64);y=y.astype(np.float64);kx=.5*np.mean(np.sum((x-x.mean(0))**2,axis=-1),axis=0);ky=.5*np.mean(np.sum((y-y.mean(0))**2,axis=-1),axis=0)
        out.append(float(np.linalg.norm(kx-ky)/max(float(np.linalg.norm(ky)),1e-16)))
    return np.array(out,dtype=np.float64)


def normalized_margin(nested_calibration,action_scales,rows):
    actions=np.asarray(nested_calibration['inner_selected_actions'],dtype=int);scales=np.asarray(action_scales,dtype=np.float64)
    if scales.shape!=(len(rows),15) or len(actions)!=len(rows) or not np.isfinite(scales).all() or (scales<0).any():raise ValueError('Invalid calibration scales')
    selected=scales[np.arange(len(rows)),actions];den=np.maximum(selected,1e-8)
    predicted=np.asarray(nested_calibration['inner_predicted_TKE_gain']);truth=np.asarray(nested_calibration['inner_true_TKE_gain'])
    cal=case_optimism_margin(predicted/den,truth/den,actions!=0,rows,.9)
    return dict(**cal,scale_definition='L2(Kcandidate-Kbaseline)/L2(Kbaseline)',numerical_scale_floor=1e-8,selected_scales=selected.tolist(),scope_note='Empirical normalized optimism, not a bound using true future TKE or a coverage guarantee')


def normalized_mean_mask(actions,point_gains,action_scales,relative_margin):
    actions=np.asarray(actions,dtype=int);p=np.asarray(point_gains);s=np.asarray(action_scales,dtype=np.float64)
    if p.shape!=(len(actions),15,3) or s.shape!=(len(actions),15) or (actions<0).any() or (actions>=15).any() or not np.isfinite(s).all() or (s<0).any() or not np.isfinite(relative_margin) or relative_margin<0:raise ValueError('Invalid inference arguments')
    ids=np.arange(len(actions));threshold=relative_margin*np.maximum(s[ids,actions],1e-8)
    return (actions!=0)&(p[ids,actions,1]>=threshold),threshold
