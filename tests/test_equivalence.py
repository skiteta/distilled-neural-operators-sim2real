"""The readable reference (dno/) computes exactly what the submitted runtime computes.

The reference is compared with the runtime code shipped in package/ (byte-identical to the code of Codabench
submission 948431, see `test_package_runtime_is_the_submitted_code`), with the TKECAL patch applied as in
package/solution.py. Weights are randomized (all parameters, including zero-initialized output layers), the same
weights are loaded into both implementations, and outputs must be bitwise equal on CPU.

Optional: with DNO_RUNTIME=/path/to/output/runtime (written by package/solution.py:train) and a CUDA device,
`test_trained_runtime_on_gpu` compares the reference with the competition runtime (CUDA graph, fused half-precision
FFN kernels) on the trained weights.
Run: python -m pytest tests -q
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "package"
RUNTIME_CODE = PACKAGE / "runs/FINAL_REBUILD/driver/runtime_code_c973h"
sys.path.insert(0, str(ROOT))
import dno  # noqa: E402
from dno.intervals import DiagnosticHead, HiddenResidualHead, IntervalHeads  # noqa: E402
from dno.model import DistilledFlowOperator, OperatorConfig  # noqa: E402

DEPLOYED = dict(
    in_steps=20,
    out_steps=20,
    width=128,
    depth=4,
    spectral_rank=32,
    modes_y=12,
    modes_x=20,
    expansion=3,
    dropout=0.03,
    use_pressure=True,
    use_physics_features=False,
)
HEAD = dict(
    diagnostic_channels=186,
    hidden_channels=96,
    out_steps=20,
    maximum_normalized_shift=0.5,
    minimum_normalized_radius=1e-4,
    maximum_normalized_radius=4.0,
    initial_log_radius=-0.9228481920249607,
)


def original_modules():
    """The runtime modules of the submission (factorized_model and its `realpde` package), isolated from dno."""
    sys.path.insert(0, str(RUNTIME_CODE))
    try:
        for name in [m for m in sys.modules if m == "realpde" or m.startswith("realpde.") or m == "factorized_model"]:
            del sys.modules[name]
        return {
            name: importlib.import_module(name)
            for name in (
                "factorized_model",
                "realpde.fast_interval",
                "realpde.sps_radius_capacity",
                "realpde.hidden_sps_interval",
                "realpde.flow_diagnostics",
            )
        }
    finally:
        sys.path.remove(str(RUNTIME_CODE))


def randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            if p.is_complex():
                noise = torch.complex(noise, torch.randn(p.shape, generator=generator)) / 10
            p.add_(0.05 * noise.to(p.dtype))


def windows(seed: int, n: int = 3, pressure: bool = False) -> torch.Tensor:
    """Spatially and temporally correlated synthetic windows in physical-like units."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, 20)[None, :, None, None, None]
    base = torch.randn(n, 1, 32, 64, 3, generator=g).cumsum(2).cumsum(3) * 0.02 + 0.3
    wave = torch.sin(6.0 * t + torch.randn(n, 1, 32, 64, 3, generator=g))
    x = base + 0.1 * wave + 0.02 * torch.randn(n, 20, 32, 64, 3, generator=g)
    if not pressure:
        x[..., 2] = 0
    return x.float()


def original_tensor_forward(mods, model, center, radius, hidden_head, observed_center):
    """`_tensor_forward` of the submitted submission.py (after the TKECAL patch), executed from its own source."""
    sys.path.insert(0, str(PACKAGE))
    import tkecal_patch

    sys.path.remove(str(PACKAGE))
    source = tkecal_patch.apply((RUNTIME_CODE / "submission.py").read_text())
    tree = ast.parse(source)
    wanted = {"_three_channels", "_env_blur", "_envelope", "_tensor_forward"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_ENV_") for t in n.targets))
        or (isinstance(n, ast.Import) and any(a.asname == "_F" for a in n.names))
    ]
    namespace = dict(
        torch=torch,
        _DEVICE=torch.device("cpu"),
        _MODEL=model,
        _CENTER_HEAD=center,
        _RADIUS_HEAD=radius,
        _HIDDEN_HEAD=hidden_head,
        _OBSERVED_CENTER_HEAD=observed_center,
        predict_with_hidden=mods["realpde.hidden_sps_interval"].predict_with_hidden,
        flow_diagnostics=mods["realpde.flow_diagnostics"].flow_diagnostics,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "submission.py", "exec"), namespace)
    assert wanted <= namespace.keys()
    return namespace["_tensor_forward"]


def build_pair(seed: int = 0):
    mods = original_modules()
    fm, fi, sr, hs = (
        mods[k]
        for k in (
            "factorized_model",
            "realpde.fast_interval",
            "realpde.sps_radius_capacity",
            "realpde.hidden_sps_interval",
        )
    )
    torch.manual_seed(seed)
    model = fm.FactorizedFlowOperator(fm.ModelConfig.from_dict(DEPLOYED)).eval()
    center = fi.FastIntervalHead(fi.FastIntervalConfig.from_dict(HEAD)).eval()
    radius = sr.ExpandedFastIntervalHead(fi.FastIntervalConfig.from_dict(HEAD)).eval()
    hidden_head, observed_center = hs.HiddenIntervalResidual(128, 20).eval(), hs.HiddenIntervalResidual(168, 20).eval()
    for k, m in enumerate((model, center, radius, hidden_head, observed_center)):
        randomize(m, seed * 10 + k)
    ref_model = DistilledFlowOperator(OperatorConfig.from_checkpoint(DEPLOYED)).eval()
    ref_model.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith(dno.model.UNUSED_CHECKPOINT_PREFIXES)},
        strict=True,
    )
    heads = IntervalHeads(
        DiagnosticHead(HEAD), DiagnosticHead(HEAD, two_streams=True), HiddenResidualHead(128), HiddenResidualHead(168)
    ).eval()
    for ref, orig in (
        (heads.center, center),
        (heads.radius, radius),
        (heads.hidden, hidden_head),
        (heads.observed_center, observed_center),
    ):
        ref.load_state_dict(orig.state_dict(), strict=True)
    return mods, (model, center, radius, hidden_head, observed_center), ref_model, heads


def test_package_runtime_is_the_submitted_code():
    expected = json.loads((PACKAGE / "code_948431.json").read_text())
    for name in (
        "factorized_model.py",
        "realpde/model.py",
        "realpde/fast_interval.py",
        "realpde/sps_radius_capacity.py",
        "realpde/hidden_sps_interval.py",
        "realpde/flow_diagnostics.py",
    ):
        assert hashlib.sha256((RUNTIME_CODE / name).read_bytes()).hexdigest() == expected[name], name


@pytest.mark.parametrize("seed,pressure", [(0, False), (1, True)])
def test_point_model_is_bitwise_equal(seed, pressure):
    mods, (model, *_), ref_model, _ = build_pair(seed)
    x = windows(100 + seed, pressure=pressure)
    with torch.inference_mode():
        point, hidden = mods["realpde.hidden_sps_interval"].predict_with_hidden(model, x)
        ref_point, ref_hidden = ref_model(x)
    assert torch.equal(point, ref_point) and torch.equal(hidden, ref_hidden)
    assert not torch.equal(point[..., :2], x[:, -1:, ..., :2].expand_as(point[..., :2]))  # nontrivial forecast


@pytest.mark.parametrize("seed", [0, 1])
def test_full_inference_is_bitwise_equal(seed):
    mods, (model, center, radius, hidden_head, observed_center), ref_model, heads = build_pair(seed)
    original = original_tensor_forward(mods, model, center, radius, hidden_head, observed_center)
    predictor = dno.Predictor(ref_model, heads)
    x = windows(200 + seed)
    prediction, lower, upper, _ = original(x)
    ref_prediction, ref_lower, ref_upper = predictor.forward(x)
    assert torch.equal(prediction, ref_prediction) and torch.equal(lower, ref_lower) and torch.equal(upper, ref_upper)
    assert (upper - lower)[..., :2].min() > 0
    # numpy API with a different batch split: every window is computed independently, so results agree up to
    # float32 rounding (CPU convolution kernels may change their summation order with the batch size)
    out = predictor.predict(x.numpy())
    predictor.batch_size = 2
    split = predictor.predict(x.numpy())
    assert all(np.abs(out[k] - split[k]).max() <= 1e-5 * np.abs(out[k]).max() for k in out)


def test_parameter_count():
    _, (model, center, radius, hidden_head, observed_center), ref_model, heads = build_pair(0)
    unused = sum(p.numel() for n, p in model.named_parameters() if n.startswith(dno.model.UNUSED_CHECKPOINT_PREFIXES))
    assert sum(p.numel() for p in ref_model.parameters()) == sum(p.numel() for p in model.parameters()) - unused
    assert (
        sum(p.numel() for p in heads.parameters())
        == sum(p.numel() for m in (center, radius, hidden_head, observed_center) for p in m.parameters())
        == 151_640
    )


@pytest.mark.skipif(
    not os.environ.get("DNO_RUNTIME") or not torch.cuda.is_available(), reason="needs DNO_RUNTIME and CUDA"
)
def test_trained_runtime_on_gpu():
    runtime = Path(os.environ["DNO_RUNTIME"])
    sys.path.insert(0, str(PACKAGE))
    import solution

    competition = solution.load_predictor(runtime.parent, "cuda")
    reference = dno.Predictor.from_runtime(runtime, device="cuda")
    x = windows(300, n=10).numpy()
    a, b = competition.predict(x), reference.predict(x)
    for key in ("prediction", "lower", "upper"):
        scale = np.abs(a[key][..., :2]).max()
        assert np.abs(a[key] - b[key]).max() <= 1e-3 * scale, key
