"""Compare whole-case point constraints with window constraints at fixed weight."""
import torch
from realpde.anchored_amplitude import point_penalty


def grouped_loss(rel,tke,obj,reference_rel,reference_tke,weight,groups,case_level):
    losses=[];penalties=[]
    for ids in groups:
        reward=(weight[ids]*obj[ids]).mean()
        if case_level:
            penalty=weight[ids].mean()*point_penalty(rel[ids].mean(),tke[ids].mean(),reference_rel[ids].mean(),reference_tke[ids].mean())
        else:
            penalty=(weight[ids]*point_penalty(rel[ids],tke[ids],reference_rel[ids],reference_tke[ids])).mean()
        losses.append(-reward+penalty);penalties.append(penalty)
    return torch.stack(losses).mean(),torch.stack(penalties).mean()
