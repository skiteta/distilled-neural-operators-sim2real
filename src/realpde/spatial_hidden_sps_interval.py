"""Window-local residual intervals from a frozen point model's final hidden field."""
from __future__ import annotations
import torch
from torch import nn


class HiddenIntervalResidual(nn.Module):
    architecture = 'hidden_interval_residual'

    def __init__(self, channels=128, out_steps=20, kernel_size=3):
        super().__init__()
        self.channels, self.out_steps = channels, out_steps
        assert kernel_size in (3, 7)
        self.kernel_size = kernel_size
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels), nn.GELU(),
            nn.Conv2d(channels, out_steps * 4, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, hidden):
        raw = self.network(hidden)
        batch, _, height, width = raw.shape
        value = raw.reshape(batch, self.out_steps, 2, 2, height, width)
        shift = .25 * torch.tanh(value[:, :, 0].permute(0, 1, 3, 4, 2))
        log_radius = .50 * torch.tanh(value[:, :, 1].permute(0, 1, 3, 4, 2))
        return shift, log_radius


def predict_with_hidden(point_model, observed):
    """Capture only this call; remove the hook even if point inference fails."""
    captured = []
    def hook(module, inputs, output):
        captured.append(output.detach())
    handle = point_model.final_norm.register_forward_hook(hook)
    try:
        prediction = point_model(observed).float()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('expected one final hidden field for this call')
    return prediction, captured[0]


def load_hidden_residual(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != HiddenIntervalResidual.architecture:
        raise ValueError('checkpoint architecture mismatch')
    model = HiddenIntervalResidual(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def expand_spatial_head(original, kernel_size=7):
    """Embed the original3x3 operator in7x7; all other parameters unchanged."""
    assert original.network[2].kernel_size == (3, 3) and kernel_size == 7
    device=next(original.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(202609221069)
        expanded=HiddenIntervalResidual(original.channels,original.out_steps,kernel_size)
    old=original.state_dict();state={k:v.detach().cpu().clone() for k,v in old.items()}
    weight=state['network.2.weight'];large=weight.new_zeros((weight.shape[0],1,kernel_size,kernel_size));large[:,:,2:5,2:5]=weight
    state['network.2.weight']=large;expanded.load_state_dict(state,strict=True)
    return expanded.to(device).eval().requires_grad_(True)
