"""TKECAL inference transform of submission.py (strings verbatim from the historical builder runs/EXACTLANE/make_tkecal.py).

The point prediction p of each window is rescaled around its temporal mean m: p' = m + r (p - m), with
r = clip((G*k_obs / max(G*k_pred, 1e-12))^0.05, 0.8, 1.25); k = 0.5 * sum over u,v of the temporal variance of the
20 observed (k_obs) or predicted (k_pred) frames; G = Gaussian blur, sigma 4 px, edge-replicate. Same-window inputs
only; applied after the interval is built, so lower/upper are unchanged.
"""
ANCHOR = "@torch.inference_mode()\ndef _tensor_forward(batch):\n"
HELPERS = '# TKECAL (runs/TKECAL/rule_TKECAL.json + amendment_1, gamma 0.10; semantics of runs/TKECAL/tkecal_torch.py with the kernel cached on the device for graph capture): envelope rescaling of the point fluctuations, computed from the same window only.\n# p\' = m + r (p - m); m = temporal mean of the 20 predicted frames; r = clip((G*k_obs / max(G*k_pred, 1e-12))^(gamma/2), gamma = 0.10, 0.8, 1.25);\n# k = 0.5 * sum_{u,v} population variance over time; G = Gaussian sigma 4 px (scipy truncate 4.0 -> radius 16), edge-replicate.\n# Applied after the bounds are built from the unmodified point, so the submitted interval is unchanged.\nimport torch.nn.functional as _F\n_ENV_RADIUS = int(4.0 * 4.0 + 0.5)\n_ENV_W = torch.exp(-0.5 * (torch.arange(-_ENV_RADIUS, _ENV_RADIUS + 1, dtype=torch.float64) / 4.0) ** 2)\n_ENV_W = (_ENV_W / _ENV_W.sum()).to(device=_DEVICE, dtype=torch.float32)\n\n\ndef _env_blur(field):\n    g = field[:, None]\n    g = _F.conv2d(_F.pad(g, (_ENV_RADIUS, _ENV_RADIUS, 0, 0), mode="replicate"), _ENV_W.view(1, 1, 1, -1))\n    g = _F.conv2d(_F.pad(g, (0, 0, _ENV_RADIUS, _ENV_RADIUS), mode="replicate"), _ENV_W.view(1, 1, -1, 1))\n    return g[:, 0]\n\n\ndef _envelope(batch, point):\n    k_obs = 0.5 * batch[..., :2].var(dim=1, unbiased=False).sum(-1)\n    uv = point[..., :2]\n    mean = uv.mean(dim=1, keepdim=True)\n    dev = uv - mean\n    k_pred = 0.5 * dev.square().mean(dim=1).sum(-1)\n    ratio = (_env_blur(k_obs) / _env_blur(k_pred).clamp_min(1e-12)).pow(0.05).clamp(0.8, 1.25)\n    return torch.cat((mean + ratio[:, None, :, :, None] * dev, point[..., 2:]), dim=-1)\n\n\n'
RET_OLD = "    return point, _three_channels(center - half_width), _three_channels(center + half_width), _three_channels(center)\n"
RET_NEW = "    return _envelope(batch, point), _three_channels(center - half_width), _three_channels(center + half_width), _three_channels(center)\n"


def apply(source):
    if source.count(ANCHOR) != 1 or source.count(RET_OLD) != 1:
        raise ValueError('unexpected submission.py: TKECAL anchor or return statement not found exactly once')
    return source.replace(ANCHOR, HELPERS + ANCHOR, 1).replace(RET_OLD, RET_NEW, 1)
