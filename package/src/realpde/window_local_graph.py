"""No-padding inference with independently captured singleton CUDA graphs."""
import numpy as np
import torch


class WindowLocalGraphPredictor:
    """Each lane owns a stream, a singleton graph and permanent scratch.

    Work is grouped for transfers, while all numerical forwards remain N=1.
    Inactive lanes execute nothing. Lane contents are not retained across calls.
    """
    lane_count = 8

    def __init__(self, forward, device):
        self.forward, self.device = forward, torch.device(device)
        self.captures = []
        if self.device.type == 'cuda':
            for _ in range(self.lane_count):
                self.captures.append(self._capture())

    @torch.inference_mode()
    def _capture(self):
        x = torch.zeros((1, 20, 32, 64, 3), device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                values = self.forward(x)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values = self.forward(x)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        x.zero_()
        for v in values:
            v.zero_()
        return x, values, graph, stream

    @classmethod
    def groups(cls, count):
        while count:
            size = min(count, cls.lane_count)
            yield size
            count -= size

    @torch.inference_mode()
    def __call__(self, input_array, metadata=None):
        array = np.asarray(input_array, dtype=np.float32)
        if array.ndim != 5 or array.shape[1:] != (20, 32, 64, 3) or not len(array):
            raise ValueError('expected nonempty [N,20,32,64,3]')
        parts = [[], [], []]
        start, used = 0, 0
        try:
            for count in self.groups(len(array)):
                if self.captures:
                    current = torch.cuda.current_stream(self.device)
                    active = self.captures[:count]
                    used = max(used, count)
                    for lane, (x, _, graph, stream) in enumerate(active):
                        stream.wait_stream(current)
                        chunk = torch.from_numpy(np.ascontiguousarray(array[start+lane:start+lane+1]))
                        with torch.cuda.stream(stream):
                            x.copy_(chunk)
                            graph.replay()
                    for _, _, _, stream in active:
                        current.wait_stream(stream)
                    values = tuple(torch.cat([cap[1][k] for cap in active], 0) for k in range(3))
                else:
                    windows = [self.forward(torch.from_numpy(np.ascontiguousarray(array[start+i:start+i+1])).to(self.device)) for i in range(count)]
                    values = tuple(torch.cat([v[k] for v in windows], 0) for k in range(3))
                for bucket, value in zip(parts, values):
                    bucket.append(value.cpu().clone())
                start += count
        finally:
            if self.captures:
                current = torch.cuda.current_stream(self.device)
                for x, values, _, stream in self.captures[:used]:
                    current.wait_stream(stream)
                    x.zero_()
                    for v in values:
                        v.zero_()
        result = {k: torch.cat(p).numpy().astype(np.float32, copy=False)
                  for k, p in zip(('prediction', 'lower', 'upper'), parts)}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('nonfinite output')
        if not np.all(result['lower'] <= result['upper']):
            raise RuntimeError('invalid bounds')
        return result
