"""Capture the existing singleton graph with cuDNN TF32 disabled."""
import torch

from .window_local_graph import WindowLocalGraphPredictor


class PrecisionWindowGraphPredictor(WindowLocalGraphPredictor):
    """Change cuDNN capture precision only, restoring the caller's setting.

    Initialization, like model loading, must finish before concurrent callers
    use the process. Prediction replays captured graphs without changing flags.
    """

    def __init__(self, forward, device):
        previous = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            super().__init__(forward, device)
        finally:
            torch.backends.cudnn.allow_tf32 = previous
