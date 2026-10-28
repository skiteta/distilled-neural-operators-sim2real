"""Training-only reflection of released samples; evaluate losses in native coordinates.

Reflect height and negate v. Time and input/target boundaries remain intact.
MVPE uses fixed, non-reflection-invariant probe pixels. Therefore inverse-reflect
predictions and interval geometry before ALL training losses against native
labels. The same helper is shipped with candidates; inference has no TTA.
"""
import torch

def reflect_velocity_fields(value):
    if value.ndim != 5 or value.shape[-1] != 3:
        raise ValueError("Expected (N,T,H,W,3) with u,v,pressure")
    return value.flip(2) * value.new_tensor([1., -1., 1.])

def restore_native_geometry(prediction, shift, radius, scale, native_input, reflected):
    """Inverse map point/center vectors and unsigned widths, preserving gradients."""
    choose = reflected[:, None, None, None, None]
    native_scale = native_input[..., :2].std(dim=(1,2,3), keepdim=True).clamp_min(1e-5)
    shift_native = (shift * scale).flip(2) * shift.new_tensor([1., -1.]) / native_scale
    radius_native = (radius * scale).flip(2) / native_scale
    return (torch.where(choose, reflect_velocity_fields(prediction), prediction),
            torch.where(choose, shift_native, shift),
            torch.where(choose, radius_native, radius),
            torch.where(choose, native_scale, scale))
