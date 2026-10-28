"""A center-only feature branch; the official point and radius paths stay frozen."""
import copy
from torch import nn
from .hidden_sps_interval import predict_with_hidden


class LastBlockCenterAdapter(nn.Module):
    architecture = 'last_block_center_adapter_v1'

    def __init__(self, point_model):
        super().__init__()
        self.block = copy.deepcopy(point_model.blocks[-1])
        self.norm = copy.deepcopy(point_model.final_norm)

    def forward(self, penultimate):
        return self.norm(self.block(penultimate))


def predict_with_last_input(point_model, observed):
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach())
    handle = point_model.blocks[-1].register_forward_pre_hook(capture)
    try:
        point, hidden = predict_with_hidden(point_model, observed)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('expected exactly one last-block input from this call')
    return point, hidden, captured[0]
