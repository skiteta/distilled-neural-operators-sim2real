"""Case-excluded observable routing between two independently frozen experts."""
import numpy as np
from .stability_tke_selector import past_maps, action_features, fit_selector, predict_selector

FEATURE_NAMES=('past_proxy_per_action','action_scale','half_TKE_disagreement',
               'early_proxy_per_action','late_proxy_per_action','stability_weighted_proxy')


def observable_features(observed, anchor, candidate):
    f, scale=action_features(past_maps(observed),anchor,candidate,0.)
    return f[[1,2,3,4,5,6]],scale


def fit_fold(x, label, rows, case_folds, held_fold):
    train=np.array([i for i,r in enumerate(rows) if held_fold is None or case_folds[r['case']]!=held_fold])
    held=np.array([i for i,r in enumerate(rows) if held_fold is not None and case_folds[r['case']]==held_fold],dtype=int)
    tr=[rows[i] for i in train];tc={r['case'] for r in tr};hc={rows[i]['case'] for i in held}
    if not tc or not tc.isdisjoint(hc):raise ValueError('Nonempty case-disjoint training required')
    # Both normalizers and labels are indexed before fitting; held values cannot affect models.
    models={a:fit_selector(x[train],label[train],tr,cols) for a,cols in [('observable',range(6)),('constant',[])]}
    return models,train,held


def route(model, features):
    return predict_selector(model,features)>0


def choose(anchor, candidate, mask):
    a,b,m=np.asarray(anchor),np.asarray(candidate),np.asarray(mask)
    if a.shape!=b.shape or len(a)!=len(m) or m.dtype!=np.bool_:raise ValueError('Aligned arrays and boolean mask required')
    return np.where(m.reshape((-1,)+(1,)*(a.ndim-1)),b,a)
