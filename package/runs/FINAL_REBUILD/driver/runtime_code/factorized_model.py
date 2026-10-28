"""C679: shared features with separate mean/direction and amplitude decoders.

The amplitude decoder sees detached shared features. Its initial weights copy
the pretrained velocity decoder, so both decoded fields start equal. This is
gradient separation, not a claim of complete independence across SGD updates:
the shared features still change when the direction/mean path is trained.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from realpde.model import HybridFlowOperator, ModelConfig

SPEC = {
    'architecture': 'C688_FIXED_ENERGY_DIRECTION_RESIDUAL_V1',
    'initialization': 'C687 fixed8000 EMA; zero last-layer residual decoder',
    'trainable': 'direction_residual_head only; original trunk, amplitude, mean and buffers frozen',
    'composition': 'preserve C687 per-cell mean and joint time/uv fluctuation norm; rotate centered direction with learned residual',
    'norm_floor': 1e-8,
    'degenerate': 'if proposed centered norm <= 1e-8, use original centered field',
    'runtime': 'one frozen trunk; existing two decoders plus small direction residual decoder',
}


def factorized_velocity(direction_field, amplitude_field):
    mean = direction_field.mean(dim=1, keepdim=True)
    centered = direction_field-mean
    amplitude_centered = amplitude_field-amplitude_field.mean(dim=1, keepdim=True)
    direction_norm = torch.linalg.vector_norm(centered, dim=(1,4), keepdim=True).clamp_min(SPEC['norm_floor'])
    amplitude_norm = torch.linalg.vector_norm(amplitude_centered, dim=(1,4), keepdim=True).clamp_min(SPEC['norm_floor'])
    gain = amplitude_norm/direction_norm
    # Equivalent factorization; exact identity when the two fields are equal.
    return direction_field+(gain-1)*centered


def fixed_energy_direction(base_uv,residual):
    centered=base_uv-base_uv.mean(dim=1,keepdim=True)
    proposed=centered+residual-residual.mean(dim=1,keepdim=True)
    source_norm=torch.linalg.vector_norm(centered,dim=(1,4),keepdim=True)
    new_norm=torch.linalg.vector_norm(proposed,dim=(1,4),keepdim=True)
    scaled=proposed*(source_norm/new_norm.clamp_min(SPEC['norm_floor']))
    rotated=torch.where(new_norm>SPEC['norm_floor'],scaled,centered)
    # The difference form is exactly base_uv for the zero residual in float32.
    return base_uv+(rotated-centered)


class FactorizedFlowOperator(HybridFlowOperator):
    def __init__(self, config):
        super().__init__(config)
        # Deepcopy does not consume RNG, preserving the original loader stream.
        self.amplitude_head = copy.deepcopy(self.velocity_head)
        self.amplitude_trend_scale = nn.Parameter(self.trend_scale.detach().clone())
        self.direction_residual_head = copy.deepcopy(self.velocity_head)
        last = [m for m in self.direction_residual_head.modules() if isinstance(m,nn.Conv2d)][-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None: nn.init.zeros_(last.bias)

    def load_predecessor_state(self,state):
        result=super().load_state_dict(state,strict=False)
        expected={'direction_residual_head.'+k for k in self.direction_residual_head.state_dict()}
        assert set(result.missing_keys)==expected and not result.unexpected_keys
        self.direction_residual_head.load_state_dict(self.velocity_head.state_dict(),strict=True)
        last=[m for m in self.direction_residual_head.modules() if isinstance(m,nn.Conv2d)][-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:nn.init.zeros_(last.bias)

    def load_parent_state(self, state):
        assert not any(k.startswith('amplitude_') for k in state)
        result = super().load_state_dict(state, strict=False)
        expected = {'amplitude_trend_scale', *('amplitude_head.'+k for k in self.amplitude_head.state_dict())}
        assert set(result.missing_keys) == expected and not result.unexpected_keys
        self.amplitude_head.load_state_dict(self.velocity_head.state_dict(), strict=True)
        with torch.no_grad(): self.amplitude_trend_scale.copy_(self.trend_scale)

    def _decode(self, hidden, raw, uv_mean, uv_std, head, trend_scale):
        batch, _, height, width = hidden.shape
        correction = head(hidden).reshape(batch, self.config.out_steps, 2, height, width)
        correction = correction.permute(0,1,3,4,2)
        base_normalized = (raw[:, -1:, ..., :2]-uv_mean)/uv_std
        recent_trend = (raw[:, -1:, ..., :2]-raw[:, -5:-4, ..., :2])/(4*uv_std)
        trend = recent_trend*trend_scale[None,:,None,None,:]
        return (base_normalized+trend+correction)*uv_std+uv_mean

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
        pressure = torch.zeros_like(uv[..., :1])
        prediction = torch.cat((uv,pressure),dim=-1)
        if not return_interval: return prediction
        batch, _, height, width = hidden.shape
        raw_width = self.width_head(hidden).reshape(batch,self.config.out_steps,2,height,width)
        raw_width = raw_width.permute(0,1,3,4,2)
        half_width_uv = F.softplus(raw_width)*uv_std*self.interval_temperature.clamp_min(1e-3)
        return prediction, torch.cat((half_width_uv,torch.zeros_like(half_width_uv[..., :1])),dim=-1)


def load_factorized(path, device, *, allow_parent=False):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    model = FactorizedFlowOperator(ModelConfig.from_dict(payload['model_config'])).to(device)
    if allow_parent:
        model.load_parent_state(payload['state_dict'])
    else:
        if any(k.startswith('direction_residual_head.') for k in payload['state_dict']):model.load_state_dict(payload['state_dict'],strict=True)
        else:model.load_predecessor_state(payload['state_dict'])
    return model
