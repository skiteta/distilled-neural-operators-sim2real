from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

CASE_PATTERN = re.compile(r"(?P<re>\d+)_(?P<aoa>-?\d+(?:\.\d+)?)\.h5$")


@dataclass(frozen=True)
class CaseInfo:
    path: Path
    reynolds: float
    measured_reynolds: float
    angle: float
    frames: int
    height: int
    width: int
    has_pressure: bool

    @property
    def regime(self) -> tuple[float, float]:
        return (self.reynolds, self.angle)


def _field_group(handle: h5py.File) -> h5py.Group | h5py.File:
    if "measured_data" in handle:
        return handle["measured_data"]
    return handle


def _scalar(handle: h5py.File, names: Sequence[str], fallback: float) -> float:
    for name in names:
        if name in handle:
            return float(np.asarray(handle[name])[()])
        if name in handle.attrs:
            return float(handle.attrs[name])
    return fallback


def inspect_case(path: str | Path) -> CaseInfo:
    path = Path(path)
    match = CASE_PATTERN.search(path.name)
    fallback_re = float(match.group("re")) if match else float("nan")
    fallback_angle = float(match.group("aoa")) if match else float("nan")
    with h5py.File(path, "r") as handle:
        group = _field_group(handle)
        if "u" not in group or "v" not in group:
            raise ValueError(f"missing u/v datasets: {path}")
        frames, height, width = group["u"].shape
        if group["v"].shape != (frames, height, width):
            raise ValueError(f"u/v shape mismatch: {path}")
        measured_reynolds = _scalar(handle, ("re", "Re", "reynolds"), fallback_re)
        measured_angle = _scalar(handle, ("aoa", "AoA", "angle"), fallback_angle)
        reynolds = fallback_re if np.isfinite(fallback_re) else measured_reynolds
        angle = fallback_angle if np.isfinite(fallback_angle) else measured_angle
        return CaseInfo(
            path=path,
            reynolds=reynolds,
            measured_reynolds=measured_reynolds,
            angle=angle,
            frames=frames,
            height=height,
            width=width,
            has_pressure="p" in group,
        )


def discover_cases(
    root: str | Path, *, exclude_files: Sequence[str] = ()
) -> list[CaseInfo]:
    root = Path(root)
    if not root.exists():
        return []
    excluded = {str(name) for name in exclude_files}
    cases = [
        inspect_case(path)
        for path in sorted(root.rglob("*.h5"))
        if path.name not in excluded
    ]
    seen: set[tuple[float, float]] = set()
    for case in cases:
        if case.regime in seen:
            raise ValueError(f"duplicate regime {case.regime}: {case.path}")
        seen.add(case.regime)
    return cases


def format_inventory(cases: Iterable[CaseInfo]) -> str:
    rows = ["file\tRe_nominal\tRe_measured\tAoA\tframes\tshape\tp"]
    for case in cases:
        rows.append(
            f"{case.path.name}\t{case.reynolds:g}\t{case.measured_reynolds:g}"
            f"\t{case.angle:g}\t{case.frames}\t{case.height}x{case.width}"
            f"\t{case.has_pressure}"
        )
    return "\n".join(rows)


class FlowWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Non-copying windows over official files; performs no augmentation."""

    def __init__(
        self,
        cases: Sequence[CaseInfo],
        *,
        in_steps: int = 20,
        out_steps: int = 20,
        stride: int = 20,
        spatial_stride: int = 2,
        split: Literal["all", "train", "validation"] = "all",
        holdout_regimes: Sequence[Sequence[float]] = (),
    ) -> None:
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.horizon = self.in_steps + self.out_steps
        self.spatial_stride = int(spatial_stride)
        holdouts = {(float(item[0]), float(item[1])) for item in holdout_regimes}
        if split == "train":
            selected = [case for case in cases if case.regime not in holdouts]
        elif split == "validation":
            selected = [case for case in cases if case.regime in holdouts]
        else:
            selected = list(cases)
        self.cases = selected
        self.windows: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            self.windows.extend(
                (case_index, start)
                for start in range(0, case.frames - self.horizon + 1, int(stride))
            )
        if not self.windows:
            raise ValueError(
                f"no {split} windows; found {len(cases)} cases and holdouts={holdouts}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_index, start = self.windows[index]
        case = self.cases[case_index]
        stop = start + self.horizon
        # Competition tensors are 32x64. Avoid downsampling an already prepared file.
        step = self.spatial_stride if case.height > 32 or case.width > 64 else 1
        with h5py.File(case.path, "r") as handle:
            group = _field_group(handle)
            u = np.asarray(group["u"][start:stop:1, ::step, ::step], dtype=np.float32)
            v = np.asarray(group["v"][start:stop:1, ::step, ::step], dtype=np.float32)
            if case.has_pressure:
                p = np.asarray(group["p"][start:stop:1, ::step, ::step], dtype=np.float32)
            else:
                p = np.zeros_like(u)
        flow = np.stack((u, v, p), axis=-1)
        if flow.shape[1:3] != (32, 64):
            raise ValueError(f"expected 32x64 after subsampling, got {flow.shape}: {case.path}")
        tensor = torch.from_numpy(flow)
        return tensor[: self.in_steps], tensor[self.in_steps :]


def split_cases(
    root: str | Path, holdout_regimes: Sequence[Sequence[float]]
) -> tuple[list[CaseInfo], list[CaseInfo]]:
    cases = discover_cases(root)
    holdouts = {(float(item[0]), float(item[1])) for item in holdout_regimes}
    return (
        [case for case in cases if case.regime not in holdouts],
        [case for case in cases if case.regime in holdouts],
    )
