"""The readable training components (dno/losses.py, dno/augment.py) equal the code that trained the model.

* metrics, SPS factor, composite loss, reflection: compared with the library functions in package/.
* window sampling: compared with `phase_cadence` of the stage S15 trainer.
* the whole stage S15 objective: the `losses` closure of package/runs/TTAKD/train_simati_focus_old.py is compiled
  from its own source and run on the same tensors as a composition of dno functions; all outputs are bitwise equal.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "package"
sys.path.insert(0, str(ROOT))
from dno import augment, losses  # noqa: E402

S15_TRAINER = PACKAGE / "runs/TTAKD/train_simati_focus_old.py"


def training_library():
    """package/src/realpde (the training library), package/scripts and the reflection helper, isolated from dno."""
    paths = [str(PACKAGE / "src"), str(PACKAGE / "scripts"), str(PACKAGE / "runs/C1031R_depth_distillation")]
    sys.path[:0] = paths
    try:
        for name in [
            m
            for m in sys.modules
            if m == "realpde"
            or m.startswith("realpde.")
            or m in ("inferred_score_objective", "reflection_transform", "experiment_records")
        ]:
            del sys.modules[name]
        return {
            name: importlib.import_module(name)
            for name in (
                "realpde.metrics",
                "realpde.identity_interval_hardening",
                "inferred_score_objective",
                "reflection_transform",
            )
        }
    finally:
        for p in paths:
            sys.path.remove(p)


def flows(seed, n=4, steps=20, shape=(32, 64), zeros=False):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, steps)[None, :, None, None, None]
    x = (
        0.3
        + 0.1 * torch.sin(5 * t + torch.randn(n, 1, *shape, 3, generator=g))
        + 0.05 * torch.randn(n, steps, *shape, 3, generator=g)
    )
    x[..., 2] = 0
    if zeros:  # unmeasured PIV elements are exact zeros in the targets
        x[:, :, :3, :5, :2] = 0
    return x


def test_metrics_and_sps_factor():
    lib = training_library()
    metrics, hardening = lib["realpde.metrics"], lib["realpde.identity_interval_hardening"]
    p, t = flows(1), flows(2, zeros=True)
    assert torch.equal(losses.relative_l2(p, t), metrics.relative_l2(p, t))
    assert torch.equal(losses.tke_relative_l2(p, t), metrics.tke_relative_l2(p, t))
    assert torch.equal(losses.mvpe_relative_l2(p, t), metrics.profile_relative_l2(p, t))
    assert torch.equal(losses.accuracy_factor(p, t), hardening.sample_factor(p, t))
    assert losses.SIGMA_GLOBAL == metrics.SIGMA_GLOBAL == hardening.SIGMA_GLOBAL


def test_composite_score_loss():
    objective = training_library()["inferred_score_objective"]
    args = (torch.tensor(0.11), torch.tensor(0.52), torch.tensor(0.13), torch.tensor(0.45))
    assert torch.equal(losses.composite_score_loss(*args), objective.training_loss(*args))
    assert (losses.COMPOSITE_POWER, losses.COMPOSITE_WEIGHTS, losses.FIXED_TIME_SCORE) == (
        objective.POWER,
        objective.WEIGHTS,
        objective.FIXED_TIME,
    )


def test_reflection():
    reflection = training_library()["reflection_transform"]
    x = flows(3)
    assert torch.equal(augment.reflect(x), reflection.reflect_velocity_fields(x))
    assert torch.equal(augment.reflect(augment.reflect(x)), x)
    shift, radius, scale = torch.randn(4, 20, 32, 64, 2), torch.rand(4, 20, 32, 64, 2), torch.rand(4, 1, 1, 1, 1) + 0.1
    chosen = torch.tensor([True, False, True, False])
    ours = augment.to_native_geometry(x, shift, radius, scale, flows(4), chosen)
    theirs = reflection.restore_native_geometry(x, shift, radius, scale, flows(4), chosen)
    for a, b in zip(ours, theirs, strict=True):
        assert torch.equal(a, b)


class _Data:
    def __init__(self):
        g = torch.Generator().manual_seed(5)
        self.flows = [torch.randn(120, 64, 128, 2, generator=g), torch.randn(70, 64, 128, 2, generator=g)]
        self.windows = [(0, 3), (0, 41), (1, 10)]


def _load_trainer():
    spec = importlib.util.spec_from_file_location("s15_trainer", S15_TRAINER)
    module = importlib.util.module_from_spec(spec)
    paths = [str(PACKAGE / "src"), str(PACKAGE / "scripts")]
    sys.path[:0] = paths
    try:
        spec.loader.exec_module(module)  # defines functions only; main() is not run
    finally:
        for p in paths:
            sys.path.remove(p)
    return module


@pytest.mark.parametrize("index,phase,slow", [(0, 0, False), (1, 3, True), (2, 2, False)])
def test_window_sampling(index, phase, slow):
    data = _Data()
    x, y, _ = _load_trainer().phase_cadence(data, index, phase, int(slow))
    case, start = data.windows[index]
    rx, ry = augment.sample_window(data.flows[case], start, phase, slow)
    assert torch.equal(x, rx) and torch.equal(y, ry)


class _Forecaster(torch.nn.Module):
    """Small deterministic stand-in returning (forecast, hidden) like the point model."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mix = torch.randn(20, 20, generator=g) / 20
        self.feat = torch.randn(8, 60, generator=g) / 60

    def forward(self, x):
        forecast = torch.einsum("st,nthwc->nshwc", self.mix, x) + x[:, -1:]
        hidden = torch.einsum(
            "kc,nchw->nkhw", self.feat, x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], 60, *x.shape[2:4])
        )
        return forecast, hidden


def _interval(x, p, h):
    """Deterministic stand-in for the interval heads: (shift, radius, scale)."""
    scale = x[..., :2].std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    shift = 0.1 * torch.tanh(p[..., :2] - x[:, -1:, ..., :2])
    radius = 0.2 + 0.05 * torch.sigmoid(h.mean(1))[:, None, :, :, None].expand_as(shift)
    return shift, radius, scale


def test_stage_s15_objective_equals_trainer():
    lib = training_library()
    tree = ast.parse(S15_TRAINER.read_text())
    closure = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "losses")
    point, teacher = _Forecaster(11), _Forecaster(12)
    ikd = [0.0]
    namespace = dict(
        torch=torch,
        reflection=lib["reflection_transform"],
        M=lib["realpde.identity_interval_hardening"],
        relative_l2=lib["realpde.metrics"].relative_l2,
        tke_relative_l2=lib["realpde.metrics"].tke_relative_l2,
        profile_relative_l2=lib["realpde.metrics"].profile_relative_l2,
        training_loss=lib["inferred_score_objective"].training_loss,
        forward=lambda model, x: model(x),
        point=point,
        teacher=teacher,
        HR=False,
        SIMR=None,
        IKD=ikd,
        interval=_interval,
        interval_with=lambda heads, x, p, h: _interval(x, p, h),
        teacher_heads=None,
        m={
            "training": {
                "loss_mode": "old",
                "teacher_tta": True,
                "interval_kd": True,
                "interval_kd_center": 1.0,
                "interval_kd_radius": 0.1,
                "kd_weight": 1.0,
            }
        },
    )
    exec(compile(ast.Module(body=[closure], type_ignores=[]), str(S15_TRAINER), "exec"), namespace)
    native, target = flows(21), flows(22, zeros=True)
    selected = torch.tensor([True, False, False, True])
    total, real, point_kd, feature_kd, reward = namespace["losses"](native, target, selected, "finetune")

    # the same objective written with dno
    x = torch.where(selected[:, None, None, None, None], augment.reflect(native), native)
    forecast, hidden = point(x)
    with torch.no_grad():
        teacher_native, teacher_hidden = teacher(x)
        teacher_reflected, teacher_hidden_reflected = teacher(augment.reflect(x))
        teacher_target = 0.5 * (teacher_native + augment.reflect(teacher_reflected))
        s1, r1, c1 = _interval(x, teacher_native, teacher_hidden)
        s2, r2, c2 = _interval(augment.reflect(x), teacher_reflected, teacher_hidden_reflected)
        center_t, width_t = augment.reflection_averaged_interval(
            teacher_native[..., :2] + s1 * c1, r1 * c1, teacher_reflected[..., :2] + s2 * c2, r2 * c2
        )
    kd = losses.distillation_loss(forecast, hidden, teacher_target, teacher_hidden)
    shift, radius, scale = _interval(x, forecast, hidden)
    interval_kd = losses.interval_distillation_loss(
        forecast[..., :2] + shift * scale, radius * scale, center_t, width_t
    )
    forecast, shift, radius, scale = augment.to_native_geometry(forecast, shift, radius, scale, native, selected)
    ref_reward = losses.soft_sps_reward(forecast, target, shift, radius, scale)
    ref_real = losses.final_stage_loss(forecast, target, ref_reward)
    assert torch.equal(reward, ref_reward) and torch.equal(real, ref_real)
    assert torch.equal(point_kd + 0.1 * feature_kd, kd)
    assert torch.equal(total, ref_real + 1.0 * kd + 0.0 + interval_kd)
