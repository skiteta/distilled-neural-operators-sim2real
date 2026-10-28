"""Case-excluded calibration of optimism in a frozen trajectory selector.

The empirical case-error margin is a heuristic, not a coverage guarantee.
"""
from __future__ import annotations
import numpy as np
from .learned_trajectory_gate import case_weights,case_folds
from .point_guarded_gate import fit_point_guard,predict_point_gains,predict_sps_gains,select_guarded_actions


def fit_sps_ridge(features,sps_gains,rows):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(sps_gains,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),15) or len(rows)!=len(x) or np.any(y[:,0]!=0):raise ValueError('Invalid aligned training arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training arrays')
    w=case_weights(rows);mu=np.sum(w[:,None]*x,axis=0);sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0)).clip(1e-6)
    z=np.column_stack((np.ones(len(x)),(x-mu)/sd));pen=np.eye(z.shape[1]);pen[0,0]=0
    coef=np.linalg.solve(z.T@(w[:,None]*z)+pen,z.T@(w[:,None]*y[:,1:]))
    return dict(feature_mean=mu.tolist(),feature_scale=sd.tolist(),ridge_coefficient=coef.tolist(),train_cases=sorted({r['case'] for r in rows}))


def case_optimism_margin(predicted_gain,true_gain,active,rows,quantile=.9):
    p=np.asarray(predicted_gain,dtype=np.float64);y=np.asarray(true_gain,dtype=np.float64);active=np.asarray(active,dtype=bool)
    if p.shape!=y.shape or p.shape!=(len(rows),) or active.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid calibration arrays')
    cases={}
    for case in sorted({r['case'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['case']==case and active[i]],dtype=int)
        cases[case]=dict(selected_windows=len(ids),mean_optimism=float(np.mean(p[ids]-y[ids])) if len(ids) else 0.)
    # Each physical case has one vote, regardless of the number of correlated windows.
    margin=max(0.,float(np.quantile([v['mean_optimism'] for v in cases.values()],quantile,method='higher')))
    return dict(margin=margin,quantile=quantile,cases=cases,empty_selection_case_value=0.,scope='empirical upper case-mean optimism; not a per-window statistical guarantee')


def nested_calibration(features,sps_gains,point_gains,rows,seed=6411,quantile=.9):
    names=sorted({r['case'] for r in rows})
    if len(names)<5:raise ValueError('At least five independent cases required')
    folds=case_folds(names,5,seed);pred=np.empty(len(rows));truth=np.empty(len(rows));chosen=np.full(len(rows),-1,int);audit=[]
    for fold in range(5):
        train=np.array([i for i,r in enumerate(rows) if folds[r['case']]!=fold]);held=np.array([i for i,r in enumerate(rows) if folds[r['case']]==fold]);tr=[rows[i] for i in train];hr=[rows[i] for i in held]
        sp=fit_sps_ridge(features[train],sps_gains[train],tr);pt=fit_point_guard(features[train],point_gains[train],tr,sp['feature_mean'],sp['feature_scale']);pg=predict_point_gains(pt,features[held]);sg=predict_sps_gains(sp,features[held]);actions,_=select_guarded_actions(sg,pg,columns=(1,))
        chosen[held]=actions;pred[held]=pg[np.arange(len(held)),actions,1];truth[held]=point_gains[held,actions,1]
        assert set(sp['train_cases']).isdisjoint(r['case'] for r in hr)
        audit.append(dict(fold=fold,train_cases=sp['train_cases'],held_cases=sorted({r['case'] for r in hr}),SPS_model=sp,point_model=pt))
    assert (chosen>=0).all();cal=case_optimism_margin(pred,truth,chosen!=0,rows,quantile)
    return dict(**cal,inner_folds=folds,inner_models=audit,inner_selected_actions=chosen.tolist(),inner_predicted_TKE_gain=pred.tolist(),inner_true_TKE_gain=truth.tolist(),training_cases=names,no_hyperparameter_search=True)


def mean_branch_mask(selected_actions,predicted_point_gain,margin):
    a=np.asarray(selected_actions,dtype=int);p=np.asarray(predicted_point_gain,dtype=np.float64)
    if p.shape!=(len(a),15,3) or not np.isfinite(p).all() or not np.isfinite(margin) or margin<0 or (a<0).any() or (a>=15).any():raise ValueError('Invalid selection or calibration margin')
    return (a!=0)&(p[np.arange(len(a)),a,1]>=margin)


def choose_moment_branch(mean_prediction,moment_prediction,use_mean):
    a=np.asarray(mean_prediction);b=np.asarray(moment_prediction);mask=np.asarray(use_mean,dtype=bool)
    if a.shape!=b.shape or a.ndim!=5 or mask.shape!=(len(a),):raise ValueError('Expected matching trajectories and query mask')
    # Select a complete trajectory; do not interpolate moments across branches.
    return np.where(mask[:,None,None,None,None],a,b)
