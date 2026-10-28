"""Batched transfers around unchanged, independently captured N=1 forwards."""
import numpy as np
import torch

from .window_local_graph import WindowLocalGraphPredictor


class PipelinedWindowGraphPredictor(WindowLocalGraphPredictor):
    """Keep lane arithmetic fixed and synchronize only at bounded slab edges."""

    slab_size = 256

    @torch.inference_mode()
    def __call__(self, input_array, metadata=None):
        if not self.captures:
            return super().__call__(input_array, metadata)
        array = np.asarray(input_array, dtype=np.float32)
        if array.ndim != 5 or array.shape[1:] != (20, 32, 64, 3) or not len(array):
            raise ValueError('expected nonempty [N,20,32,64,3]')
        parts = []
        current = torch.cuda.current_stream(self.device)
        used = 0
        try:
            for start in range(0, len(array), self.slab_size):
                host = torch.from_numpy(np.ascontiguousarray(array[start:start+self.slab_size]))
                inputs = host.to(self.device)
                outputs = torch.empty((3, *inputs.shape), device=self.device, dtype=torch.float32)
                active_count = min(len(inputs), self.lane_count)
                used = max(used, active_count)
                for _, _, _, stream in self.captures[:active_count]:
                    stream.wait_stream(current)
                for i in range(len(inputs)):
                    x, values, graph, stream = self.captures[i % self.lane_count]
                    with torch.cuda.stream(stream):
                        x.copy_(inputs[i:i+1], non_blocking=True)
                        graph.replay()
                        for k, value in enumerate(values):
                            outputs[k, i:i+1].copy_(value, non_blocking=True)
                for _, _, _, stream in self.captures[:active_count]:
                    current.wait_stream(stream)
                # One blocking transfer keeps slab storage alive until all
                # lane reads/writes finish. No host synchronization per lane.
                parts.append(outputs.cpu())
        finally:
            for x, values, _, stream in self.captures[:used]:
                current.wait_stream(stream)
                x.zero_()
                for value in values:
                    value.zero_()
        output = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        result = {key: output[k].numpy() for k, key in enumerate(('prediction', 'lower', 'upper'))}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('nonfinite output')
        if not np.all(result['lower'] <= result['upper']):
            raise RuntimeError('invalid bounds')
        return result
