"""Average the original training loss over both vertical orientations.

Auxiliary preparation for a future trial; current C873/C877 do not import this.
The caller accumulates both half-loss gradients, then clips/steps ONCE. Deployment
keeps one model. This doubles training forward/backward views per source batch.
"""
import torch
from . import identity_interval_hardening as M
from .metrics import relative_l2, tke_relative_l2, profile_relative_l2

def native_objective(pred, shift, radius, scale, target, training):
    l2=relative_l2(pred,target)
    tke=tke_relative_l2(pred,target)
    mvpe=profile_relative_l2(pred,target)
    early=relative_l2(pred[:,:3],target[:,:3])
    factor=M.sample_factor(pred,target)
    error=(target[...,:2]-pred[...,:2])/scale
    mask=target[...,:2]!=0
    tau=training['tau_fixed']
    reward=(factor[:,None,None,None,None]*torch.sigmoid((radius-(error-shift).abs())/tau)*torch.exp(-2*radius*scale/M.SIGMA_GLOBAL))[mask].mean()
    pc=pred[...,:2]-pred[...,:2].mean(1,keepdim=True)
    tc=target[...,:2]-target[...,:2].mean(1,keepdim=True)
    pn=torch.linalg.vector_norm(pc,dim=(1,4),keepdim=True)
    tn=torch.linalg.vector_norm(tc,dim=(1,4),keepdim=True)
    wave=relative_l2(pc*(tn/pn.clamp_min(1e-8)),tc) if training['phase_energy_normalized'] else relative_l2(pred-pred.mean(1,keepdim=True),target-target.mean(1,keepdim=True))
    late=relative_l2(pred[:,10:],target[:,10:])
    loss=l2+1.*tke+.1*mvpe+.25*early-training['sps_weight']*reward+training['wave_weight']*wave+training['late_weight']*late
    return loss,dict(rel_l2=l2,tke=tke,mvpe=mvpe,early3=early,energy_normalized_phase=wave,late10=late,soft_SPS_reward=reward)

def paired_losses(forward,native_input,target,first_mask,training,reflect,restore):
    """Yield two weighted losses so backward can free each graph before the next."""
    if not training['reflection_active']:
        raise ValueError('Paired reflection requires the reflected training recipe.')
    first_mask=first_mask.to(device=native_input.device,dtype=torch.bool)
    if first_mask.shape!=(len(native_input),):
        raise ValueError('One orientation bit per source window is required.')
    mirrored=reflect(native_input)
    for view,selected in enumerate((first_mask,~first_mask)):
        x=torch.where(selected[:,None,None,None,None],mirrored,native_input)
        pred,shift,radius,scale,_=forward(x)
        pred,shift,radius,scale=restore(pred,shift,radius,scale,native_input,selected)
        loss,terms=native_objective(pred,shift,radius,scale,target,training)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite paired-view loss.')
        yield .5*loss,dict(view=view,reflected_samples=int(selected.sum()),unweighted_loss=loss.detach(),**{k:v.detach() for k,v in terms.items()})
