"""Scoped cuDNN precision setting for frozen interval-head graph capture."""
from contextlib import contextmanager
import torch


@contextmanager
def interval_cudnn_precision():
    """Disable cuDNN TF32 for the enclosed interval operations, then restore.

    This context is used during single-threaded graph initialization; graph
    replay does not enter it or change process-wide precision flags.
    """
    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous
