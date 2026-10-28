"""Observed-window autocorrelation as an inductive bias, not new information.

The eight added channels expose local temporal coherence to the point backbone.
They use only the supplied20 observations and no fitted preprocessing. The new
lift columns start at zero, preserving the inherited predictor up to rounding.
"""
from factorized_base import *

LAGS=(1,2,4,8)
class FactorizedFlowOperator(FactorizedFlowOperator):
    def __init__(self,config):
        super().__init__(config)
        old=self.lift
        assert isinstance(old,nn.Conv2d) and old.kernel_size==(1,1) and old.groups==1
        new=nn.Conv2d(old.in_channels+2*len(LAGS),old.out_channels,1,bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:old.in_channels].copy_(old.weight)
            if old.bias is not None:new.bias.copy_(old.bias)
        self.lift=new

    def _features(self,flow):
        features,mean,std=super()._features(flow)
        fluct=(flow[...,:2]-flow[...,:2].mean(dim=1,keepdim=True))/std
        correlations=[]
        for lag in LAGS:
            a,b=fluct[:,:-lag],fluct[:,lag:]
            cross=(a*b).mean(dim=1)
            norm=(a.square().mean(dim=1)*b.square().mean(dim=1)).sqrt()
            correlations.append((cross/(norm+1e-4)).permute(0,3,1,2))
        return torch.cat([features,*correlations],dim=1),mean,std


def load_factorized(path,device,*,allow_parent=False):
    assert not allow_parent,'Expanded inputs require an explicitly embedded checkpoint.'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=FactorizedFlowOperator(ModelConfig.from_dict(payload['model_config'])).to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model
