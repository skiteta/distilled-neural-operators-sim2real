"""Function-preserving interval geometry inputs to a frozen-source hidden head."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from realpde.hidden_sps_interval import HiddenIntervalResidual

class GeometryHiddenIntervalResidual(HiddenIntervalResidual):
    architecture = 'geometry_hidden_interval_residual'

    def __init__(self, channels=128, out_steps=20):
        super().__init__(channels, out_steps)
        self.geometry_input = nn.Conv2d(out_steps * 4, channels, 1, bias=False)
        nn.init.zeros_(self.geometry_input.weight)

    @classmethod
    def from_hidden(cls, source):
        result = cls(source.channels, source.out_steps)
        result.network = copy.deepcopy(source.network)
        return result

    def forward(self, hidden, base_shift, base_radius, mode='aligned'):
        n, _, h, w = hidden.shape
        center = (base_shift / .5).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)
        width = (base_radius.clamp_min(1e-4).log() / abs(math.log(1e-4))).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)
        geometry = torch.cat((center, width), dim=1)
        if mode == 'spatial_roll':
            geometry = torch.roll(geometry, shifts=(h // 2, w // 2), dims=(2, 3))
        elif mode == 'hidden_only':
            geometry = torch.zeros_like(geometry)
        elif mode != 'aligned':
            raise ValueError('unsupported geometry control')
        value = self.network[0](hidden) + self.geometry_input(geometry)
        for layer in self.network[1:]:
            value = layer(value)
        value = value.reshape(n, self.out_steps, 2, 2, h, w)
        return (.25 * torch.tanh(value[:, :, 0].permute(0, 1, 3, 4, 2)),
                .50 * torch.tanh(value[:, :, 1].permute(0, 1, 3, 4, 2)))

def load_geometry_hidden_residual(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != GeometryHiddenIntervalResidual.architecture:
        raise ValueError('checkpoint architecture mismatch')
    model = GeometryHiddenIntervalResidual(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False)
