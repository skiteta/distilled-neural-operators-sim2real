"""Window-local continuous residual mixtures and expected-SPS interval selection.

No fitting, data access, metadata or cross-call state occurs in this module.
A weighted density loss can fit q(e|x) proportional to p(e|x)*official_factor;
maximizing its interval mass times the width penalty then has the same argmax
as the conditional expected reward (the unknown normalizer cancels).
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalLogisticInterval(nn.Module):
    architecture = 'conditional_logistic_interval_v1'

    def __init__(self, hidden_channels=128, out_steps=20, components=3, width=64):
        super().__init__()
        if components not in (1, 3):
            raise ValueError('Predeclared one- or three-component comparison only')
        self.model_config = dict(hidden_channels=hidden_channels, out_steps=out_steps,
                                 components=components, width=width)
        self.components, self.out_steps = components, out_steps
        channels = hidden_channels + out_steps * 2 * 3
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GELU(),
            nn.Conv2d(width, out_steps * 2 * components * 3, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        offsets = [0.] if components == 1 else [-1., 0., 1.]
        prior = [1.] if components == 1 else [.25, .5, .25]
        self.register_buffer('offsets', torch.tensor(offsets))
        self.register_buffer('log_prior', torch.tensor(prior).log())
        self.initial_scale_bias = math.log(math.expm1(.48))

    @staticmethod
    def planes(value):
        return value.permute(0, 1, 4, 2, 3).flatten(1, 2)

    def forward(self, hidden, observed_normalized, base_shift, base_radius):
        radius = base_radius.clamp_min(1e-5)
        features = torch.cat((hidden, observed_normalized,
                              self.planes(base_shift), self.planes(radius.log())), 1)
        raw = self.network(features)
        n, _, h, w = raw.shape
        raw = raw.reshape(n, self.out_steps, 2, self.components, 3, h, w).permute(0, 1, 5, 6, 2, 3, 4)
        means = base_shift[..., None] + radius[..., None] * (self.offsets + raw[..., 0])
        scales = radius[..., None] * (.02 + F.softplus(self.initial_scale_bias + raw[..., 1]))
        log_weights = F.log_softmax(self.log_prior + raw[..., 2], dim=-1)
        return means, scales, log_weights


def weighted_logistic_nll(residual, means, scales, log_weights, mask, sample_factor):
    """Proper weighted marginal log loss; labels/weights belong to the fit split."""
    z = (residual[..., None] - means) / scales
    log_pdf = -F.softplus(z) - F.softplus(-z) - scales.log()
    nll = -torch.logsumexp(log_weights + log_pdf, dim=-1)
    weights = sample_factor[:, None, None, None, None].expand_as(nll)
    # Fixed available-pixel count: do not cancel target-dependent factor weights.
    # Missing targets contribute zero; the dataset-wide normalization is constant.
    return (nll * weights)[mask].sum() / nll.numel()


def logistic_optimal_radius(scales, physical_scale, sigma_global):
    """Exact single-logistic optimum of covered mass * exp(-2*r*u/SIGMA)."""
    return scales * torch.asinh(sigma_global / (2 * scales * physical_scale))


def mixture_interval_reward(center, radius, means, scales, log_weights,
                            physical_scale, sigma_global):
    lo = (center[..., None] - radius[..., None] - means) / scales
    hi = (center[..., None] + radius[..., None] - means) / scales
    mass = ((torch.sigmoid(hi) - torch.sigmoid(lo)).clamp_min(0) * log_weights.exp()).sum(-1)
    return mass * torch.exp(-2 * radius * physical_scale / sigma_global)


def choose_interval(means, scales, log_weights, base_shift, base_radius,
                    physical_scale, sigma_global):
    """Fixed small candidate set; baseline included, with strict-improvement ties.

    The guarantee is on MODEL-EXPECTED reward, never on unseen actual outcomes.
    No target, case ID or fitted validation threshold enters this decision.
    """
    best_center, best_radius = base_shift, base_radius
    best_reward = mixture_interval_reward(base_shift, base_radius, means, scales,
                                         log_weights, physical_scale, sigma_global)
    for index in range(means.shape[-1]):
        center = means[..., index]
        radius = logistic_optimal_radius(scales[..., index], physical_scale, sigma_global)
        reward = mixture_interval_reward(center, radius, means, scales, log_weights,
                                         physical_scale, sigma_global)
        select = reward > best_reward
        best_center = torch.where(select, center, best_center)
        best_radius = torch.where(select, radius, best_radius)
        best_reward = torch.maximum(best_reward, reward)
    if means.shape[-1] > 1:
        weights = log_weights.exp()
        center = (weights * means).sum(-1)
        equivalent_scale = (weights * (scales.square() + (3 / math.pi**2) *
                                      (means - center[..., None]).square())).sum(-1).clamp_min(1e-12).sqrt()
        radius = logistic_optimal_radius(equivalent_scale, physical_scale, sigma_global)
        reward = mixture_interval_reward(center, radius, means, scales, log_weights,
                                         physical_scale, sigma_global)
        select = reward > best_reward
        best_center = torch.where(select, center, best_center)
        best_radius = torch.where(select, radius, best_radius)
    return best_center, best_radius


def load_conditional_logistic(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('architecture') != ConditionalLogisticInterval.architecture:
        raise ValueError('Wrong residual-density checkpoint architecture')
    model = ConditionalLogisticInterval(**payload['model_config'])
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False)
