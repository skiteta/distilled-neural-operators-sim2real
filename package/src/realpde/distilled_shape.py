"""Small C665 student; original feature geometry and frozen C659 anchor retained."""
from torch import nn
from .scaled_shape_mixture import ScaledShapeMixture, FieldExpert

class DistilledShape(ScaledShapeMixture):
    def __init__(self, anchor_state, width=16):
        super().__init__('shared', anchor_state)
        self.experts = nn.ModuleList([FieldExpert(width)])
        self.width = width
