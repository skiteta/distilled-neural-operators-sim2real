"""End-to-end inference in plain PyTorch: forecast, prediction intervals and TKECAL.

    predictor = Predictor.from_runtime('/path/to/output/runtime', device='cuda')
    out = predictor.predict(x)   # x: float32 (N, 20, H, W, 3) in physical units, any N
    out['prediction'], out['lower'], out['upper']   # float32 (N, 20, H, W, 3); pressure channel is 0

`/path/to/output/runtime` is written by `package/solution.py:train()`. The competition runtime in that directory is the
same computation with speed-oriented engineering (CUDA graph capture, fused NVRTC kernels with half-precision FFN
convolutions, pinned asynchronous transfers); this module is the readable reference. tests/ checks that both agree.
"""

from __future__ import annotations

import numpy as np
import torch

from .intervals import IntervalHeads, load_interval_heads
from .model import DistilledFlowOperator, load_point_model
from .tkecal import gaussian_kernel, tke_calibrate


class Predictor:
    def __init__(self, model: DistilledFlowOperator, heads: IntervalHeads, device="cpu", batch_size: int = 32) -> None:
        self.model, self.heads = model.to(device), heads.to(device)
        self.device, self.batch_size = torch.device(device), batch_size
        self.kernel = gaussian_kernel(self.device)

    @classmethod
    def from_runtime(cls, runtime_dir, device="cpu", batch_size: int = 32) -> Predictor:
        return cls(load_point_model(f"{runtime_dir}/model.pt"), load_interval_heads(runtime_dir), device, batch_size)

    @torch.inference_mode()
    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point, hidden = self.model(window)
        lower, upper = self.heads(window, point, hidden)
        zeros = torch.zeros_like(lower[..., :1])
        return tke_calibrate(window, point, self.kernel), torch.cat((lower, zeros), -1), torch.cat((upper, zeros), -1)

    def predict(self, input_array) -> dict[str, np.ndarray]:
        x = np.asarray(input_array, dtype=np.float32)
        if x.ndim != 5 or x.shape[1] != 20 or x.shape[-1] != 3:
            raise ValueError(f"expected (N, 20, H, W, 3), got {x.shape}")
        parts = [
            self.forward(torch.from_numpy(np.ascontiguousarray(x[i : i + self.batch_size])).to(self.device))
            for i in range(0, len(x), self.batch_size)
        ]
        return {
            key: torch.cat([p[k] for p in parts]).cpu().numpy()
            for k, key in enumerate(("prediction", "lower", "upper"))
        }
