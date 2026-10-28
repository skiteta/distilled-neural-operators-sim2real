"""Reuse C665 student's identical input statistics and baseline moment terms.

Only per-window arithmetic is shared. Frozen parameters and FP64 moment
projection order are retained; no previous-call prediction cache is used.
"""
import torch

def shared_shape(head, observed, baseline):
    om = observed.mean(1, keepdim=True)
    bm = baseline.mean(1, keepdim=True)
    fo, fb = observed - om, baseline - bm
    norm = (.5 * (fo.square().mean((1,2,3,4), keepdim=True)
                 + fb.square().mean((1,2,3,4), keepdim=True))).sqrt().clamp_min(1e-6)
    rms = fb.square().sum(-1, keepdim=True).mean(1, keepdim=True).sqrt()
    pack = lambda z: z.permute(0,1,4,2,3).flatten(1,2)
    normalized_o, normalized_b = pack(fo/norm), pack(fb/norm)
    b64 = baseline.double()
    mean64 = b64.mean(1, keepdim=True)
    fluct64 = b64 - mean64
    energy64 = fluct64.square().sum(-1, keepdim=True).mean(1, keepdim=True)
    def project(value):
        a = value.double()
        fa = a - a.mean(1, keepdim=True)
        ea = fa.square().sum(-1, keepdim=True).mean(1, keepdim=True)
        scale = (energy64.clamp_min(1e-30)/ea.clamp_min(1e-30)).sqrt()
        return (mean64 + torch.where((ea>1e-30)&(energy64>1e-30), fa*scale, fluct64)).to(baseline.dtype)
    z = head.anchor.net(torch.cat((normalized_o, normalized_b),1))
    z = z.reshape(len(z),20,2,z.shape[2],z.shape[3]).permute(0,1,3,4,2)
    anchor = project(baseline + .1*torch.tanh(z)*rms)
    h,w=baseline.shape[2:4]
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=baseline.device),torch.linspace(-1,1,w,device=baseline.device),indexing='ij')
    coords=torch.stack((yy,xx))[None].expand(len(baseline),-1,-1,-1)
    features=torch.cat((pack(observed/norm),normalized_o,normalized_b,pack(bm/norm),coords),1)
    z=head.experts[0](features)
    z=z.reshape(len(z),20,2,z.shape[2],z.shape[3]).permute(0,1,3,4,2)
    return project(anchor + .2*z.tanh()*rms)
