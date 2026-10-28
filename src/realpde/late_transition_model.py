"""Three shared late-horizon transitions with exact early-frame retention.

Uses observed-window backbone context and its own predicted future only.
For each fixed base trajectory it preserves the first5 frames, whole-window
mean and joint time/uv energy (up to floating-point rounding). Fine-tuning the
shared backbone can still change these quantities relative to the old model.
"""
from factorized_base import *

class LateTransition(nn.Module):
    def __init__(self, context_width):
        super().__init__()
        width=32
        self.initial=nn.Sequential(nn.Conv2d(context_width,width,1),nn.Tanh())
        self.source=nn.Sequential(nn.Conv2d(21,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.GELU())
        self.gates=nn.Conv2d(2*width,2*width,3,padding=1)
        self.candidate=nn.Conv2d(2*width,width,3,padding=1)
        self.norm=nn.GroupNorm(8,width)
        self.readout=nn.Conv2d(width,10,1)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self, hidden, base, mean, std):
        batch,_,height,width,_=base.shape
        state=self.initial(F.avg_pool2d(hidden,2))
        normalized=(base-mean)/std
        previous=normalized[:,:5]
        corrections=[]
        for index in range(3):
            proposal=normalized[:,5+5*index:10+5*index]
            a=proposal.permute(0,1,4,2,3).reshape(batch,10,height,width)
            b=previous.permute(0,1,4,2,3).reshape(batch,10,height,width)
            time=torch.full_like(a[:,:1],(index+1)/3.)
            source=self.source(torch.cat((a,b,time),1))
            reset,update=self.gates(torch.cat((source,state),1)).chunk(2,1)
            reset,update=reset.sigmoid(),update.sigmoid()
            candidate=torch.tanh(self.candidate(torch.cat((source,reset*state),1)))
            state=torch.lerp(state,candidate,update)
            delta=F.interpolate(self.readout(self.norm(state)),size=(height,width),mode='bilinear',align_corners=False)
            delta=.35*torch.tanh(delta.reshape(batch,5,2,height,width).permute(0,1,3,4,2))
            corrections.append(delta*std)
            previous=proposal+delta
        tail=fixed_energy_direction(base[:,5:],torch.cat(corrections,1))
        return torch.cat((base[:,:5],tail),1)

class FactorizedFlowOperator(FactorizedFlowOperator):
    def __init__(self, config):
        super().__init__(config)
        # New parameters must not change the inherited CPU/CUDA RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(202609231100)
            self._late_transition=LateTransition(config.width)

    def forward(self, flow, *, return_interval=False):
        raw = flow
        features, uv_mean, uv_std = self._features(raw)
        hidden = self.lift(features)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index+1 == max(1,len(self.blocks)//2): hidden = self.multiscale(hidden)
        hidden = self.final_norm(hidden)
        direction = self._decode(hidden, raw, uv_mean, uv_std, self.velocity_head, self.trend_scale)
        amplitude = self._decode(hidden.detach(), raw, uv_mean, uv_std, self.amplitude_head, self.amplitude_trend_scale)
        base_uv = factorized_velocity(direction, amplitude)
        batch, _, height, width = hidden.shape
        residual = self.direction_residual_head(hidden.detach()).reshape(batch,self.config.out_steps,2,height,width).permute(0,1,3,4,2)*uv_std
        uv = fixed_energy_direction(base_uv,residual)
        uv = self._late_transition(hidden, uv, uv_mean, uv_std)
        pressure = torch.zeros_like(uv[..., :1])
        prediction = torch.cat((uv,pressure),dim=-1)
        if not return_interval: return prediction
        batch, _, height, width = hidden.shape
        raw_width = self.width_head(hidden).reshape(batch,self.config.out_steps,2,height,width)
        raw_width = raw_width.permute(0,1,3,4,2)
        half_width_uv = F.softplus(raw_width)*uv_std*self.interval_temperature.clamp_min(1e-3)
        return prediction, torch.cat((half_width_uv,torch.zeros_like(half_width_uv[..., :1])),dim=-1)


def load_factorized(path,device,*,allow_parent=False):
    assert not allow_parent,'Use an explicitly embedded checkpoint.'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=FactorizedFlowOperator(ModelConfig.from_dict(payload['model_config'])).to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model
