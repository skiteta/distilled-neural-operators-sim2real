"""Train-only ridge point-error guard for frozen trajectory mixtures."""
from __future__ import annotations
import numpy as np
from .learned_trajectory_gate import case_weights


def fit_point_guard(features, error_gains, rows, mean, scale):
    """Fit raw error improvements: baseline error minus candidate error.

    Normalizer is inherited from the same training split's frozen SPS ridge.
    No held features or targets are accepted by this fitting API.
    """
    x=np.asarray(features,dtype=np.float64);y=np.asarray(error_gains,dtype=np.float64)
    mean=np.asarray(mean,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),15,3) or len(rows)!=len(x):
        raise ValueError('Expected aligned features, rows, and N,15,3 error gains')
    if mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or (scale<=0).any():
        raise ValueError('Invalid training normalizer')
    if not all(np.isfinite(v).all() for v in (x,y,mean,scale)) or np.any(y[:,0]!=0):
        raise ValueError('Nonfinite inputs or nonzero baseline gain')
    w=case_weights(rows);z=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    penalty=np.eye(z.shape[1]);penalty[0,0]=0
    coef=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w[:,None]*y[:,1:].reshape(len(x),42)))
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficient=coef.tolist(),ridge_lambda=1.,
                train_cases=sorted({r['case'] for r in rows}),train_rows=len(rows))


def predict_point_gains(model, features):
    x=np.asarray(features,dtype=np.float64);mu=np.asarray(model['mean']);sd=np.asarray(model['scale']);coef=np.asarray(model['coefficient'])
    if x.ndim!=2 or x.shape[1:]!=mu.shape or not np.isfinite(x).all():
        raise ValueError('Invalid query features')
    # Fixed per-query arithmetic provides exact batching and ordering invariance.
    values=np.array([np.r_[1.,(row-mu)/sd]@coef for row in x],dtype=np.float64).reshape(len(x),14,3)
    return np.concatenate((np.zeros((len(x),1,3)),values),axis=1)


def predict_sps_gains(metadata,features):
    x=np.asarray(features,dtype=np.float64);mu=np.asarray(metadata['feature_mean']);sd=np.asarray(metadata['feature_scale']);coef=np.asarray(metadata['ridge_coefficient'])
    values=np.array([np.r_[1.,(row-mu)/sd]@coef for row in x],dtype=np.float64).reshape(len(x),14)
    return np.column_stack((np.zeros(len(x)),values))


def select_guarded_actions(sps_gains,point_gains,columns=(0,1,2),allowed=None):
    """Maximize predicted SPS subject to predicted nonnegative raw error gain.

    This is a learned prediction of non-degradation, not a statistical guarantee.
    """
    sps=np.asarray(sps_gains);point=np.asarray(point_gains)
    if sps.ndim!=2 or sps.shape[1]!=15 or point.shape!=(*sps.shape,3):
        raise ValueError('Expected N,15 SPS and N,15,3 point gains')
    if not np.isfinite(sps).all() or not np.isfinite(point).all() or np.any(sps[:,0]!=0) or np.any(point[:,0]!=0):
        raise ValueError('Nonfinite predictions or invalid baseline')
    feasible=(point[:,:,list(columns)]>=0).all(-1)
    if allowed is not None:
        mask=np.zeros(15,bool);mask[np.asarray(allowed,dtype=int)]=True;mask[0]=True;feasible &= mask
    feasible[:,0]=True
    return np.argmax(np.where(feasible,sps,-np.inf),axis=1),feasible
