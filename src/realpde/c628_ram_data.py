"""Bounded, tensor-identical RAM reads for the fixed C628 real-data recipe."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import _field_group


class InMemoryWindows(Dataset):
    """Retain the original case/window order and return independent sample storage."""

    def __init__(self, source, *, return_index=False, max_bytes=6 * 1024**3):
        self.source = source
        self.cases = source.cases
        self.windows = source.windows
        self.in_steps = getattr(source, 'in_steps', 20)
        self.out_steps = getattr(source, 'out_steps', 20)
        self.return_index = return_index
        self.cache_bytes = sum(c.frames * 32 * 64 * 3 * 4 for c in self.cases)
        if self.cache_bytes > max_bytes:
            raise MemoryError('real RAM data exceeds explicit6GiB cache bound')
        self.flows = []
        for case in self.cases:
            stride = getattr(source, 'spatial_stride', 2) if case.height > 32 or case.width > 64 else 1
            flow = np.empty((case.frames, 32, 64, 3), dtype=np.float32)
            with h5py.File(case.path, 'r') as handle:
                group = _field_group(handle)
                for channel, field in enumerate(('u', 'v', 'p')):
                    if field == 'p' and not case.has_pressure:
                        flow[..., channel] = 0
                    else:
                        value = np.asarray(group[field][:, ::stride, ::stride], dtype=np.float32)
                        if value.shape != flow.shape[:-1]:
                            raise ValueError('unexpected RAM field shape: ' + str(case.path))
                        flow[..., channel] = value
            self.flows.append(torch.from_numpy(flow))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        case, start = self.windows[index]
        flow = self.flows[case][start:start + self.in_steps + self.out_steps].clone()
        values = (flow[:self.in_steps], flow[self.in_steps:])
        if self.return_index:
            return (*values, torch.tensor(index, dtype=torch.long))
        return values


def install_real_read_cache(base):
    """Called only for real stages; sim pretraining remains the original path."""
    original_flow = base.flow_dataset
    original_native = base.Native79Dataset

    def flow_dataset(*args, **kwargs):
        return InMemoryWindows(original_flow(*args, **kwargs))

    def native_dataset(cases):
        return InMemoryWindows(original_native(cases), return_index=True)

    base.flow_dataset = flow_dataset
    base.Native79Dataset = native_dataset
