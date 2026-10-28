"""Released native windows with selectable starts and unchanged 79-frame support."""
from __future__ import annotations
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import _field_group

class ReleasedTemporalStartDataset(Dataset):
    def __init__(self, cases, stride=1, budget=None):
        if stride < 1:
            raise ValueError('stride must be positive')
        self.cases = list(cases)
        self.stride = stride
        self.flows = []
        self.windows = []
        for ci, case in enumerate(self.cases):
            spacing = 2 if case.height > 32 or case.width > 64 else 1
            with h5py.File(case.path, 'r') as handle:
                group = _field_group(handle)
                u = np.asarray(group['u'][:, ::spacing, ::spacing], dtype=np.float32)
                v = np.asarray(group['v'][:, ::spacing, ::spacing], dtype=np.float32)
                p = np.asarray(group['p'][:, ::spacing, ::spacing], dtype=np.float32) if case.has_pressure else np.zeros_like(u)
            flow = torch.from_numpy(np.stack((u, v, p), axis=-1))
            if tuple(flow.shape) != (case.frames, 32, 64, 3) or not torch.isfinite(flow).all():
                raise ValueError('released trajectory shape or finiteness mismatch')
            self.flows.append(flow)
            self.windows.extend((ci, start) for start in range(0, case.frames - 79 + 1, stride))
            if budget is not None:
                budget.check()
        if not self.windows:
            raise ValueError('no valid released windows')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        case, start = self.windows[index]
        native = self.flows[case][start:start + 40]
        return native[:20], native[20:], torch.tensor(index, dtype=torch.long)

    @property
    def storage_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.flows)
