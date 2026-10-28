"""Warm-start temporal fluctuation adaptation with an unchanged mean field."""
import copy
from torch import nn
from .last_block_center_adapter import LastBlockCenterAdapter

class MeanPreservingDecoderAdapter(nn.Module):
    architecture = 'mean_preserving_decoder_adapter_v1'
    def __init__(self, point_model, adapt_features=True):
        super().__init__()
        self.adapt_features = adapt_features
        self.feature = LastBlockCenterAdapter(point_model) if adapt_features else None
        self.decoder = copy.deepcopy(point_model.velocity_head)
        self.out_steps = point_model.config.out_steps

    def forward(self, penultimate, hidden, reference_correction, uv_std):
        features = self.feature(penultimate) if self.feature is not None else hidden
        raw = self.decoder(features)
        n, _, height, width = raw.shape
        correction = raw.reshape(n, self.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        delta = (correction - reference_correction) * uv_std
        return delta - delta.mean(dim=1, keepdim=True)
