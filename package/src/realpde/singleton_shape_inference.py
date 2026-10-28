"""Replay the original one-window arithmetic, with a shared frozen anchor.

Graph buffers are scratch space, fully overwritten per window. No previous
window or previous call affects outputs; no prediction cache or adaptive state.
"""
import numpy as np
import torch
from .trajectory_moment_projection import restore_baseline_moments

class SingletonGraphEnsemble:
    @torch.inference_mode()
    def __init__(self, heads):
        if not heads:
            raise ValueError('empty ensemble')
        self.heads = tuple(heads)
        self.device = next(heads[0].parameters()).device
        anchor = heads[0].anchor.state_dict()
        if not all(all(torch.equal(v, anchor[k]) for k, v in h.anchor.state_dict().items()) for h in heads):
            raise ValueError('shared anchor must be identical')
        self.graph = None
        if self.device.type == 'cuda':
            self.x = torch.zeros((1,20,32,64,2), device=self.device)
            self.b = torch.zeros_like(self.x)
            # Warm up on an auxiliary stream before capture, using no data.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
                for _ in range(3):
                    self.mean = self._forward(self.x, self.b)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
                with torch.cuda.graph(self.graph):
                    self.mean = self._forward(self.x, self.b)

    def _forward(self, x, b):
        anchor = self.heads[0].anchor(x, b)
        return sum(head(x, b, anchor).double() for head in self.heads) / len(self.heads)

    @torch.inference_mode()
    def __call__(self, observed, baseline):
        if len(observed) != len(baseline):
            raise ValueError('length mismatch')
        if not len(baseline):
            return np.empty_like(baseline)
        x = torch.from_numpy(np.ascontiguousarray(observed, dtype=np.float32)).to(self.device)
        b = torch.from_numpy(np.ascontiguousarray(baseline, dtype=np.float32)).to(self.device)
        result = torch.empty(b.shape, dtype=torch.float64, device=self.device)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
            for i in range(len(b)):
                if self.graph is not None:
                    self.x.copy_(x[i:i+1]); self.b.copy_(b[i:i+1])
                    self.graph.replay()
                    result[i:i+1].copy_(self.mean)
                else:
                    result[i:i+1].copy_(self._forward(x[i:i+1], b[i:i+1]))
        value = restore_baseline_moments(result.cpu().numpy(), baseline)
        if self.graph is not None:
            self.x.zero_(); self.b.zero_(); self.mean.zero_()
        return value
