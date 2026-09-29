"""Mixed-precision policy helpers for network execution."""

from __future__ import annotations

from typing import Literal

import torch as T


def resolve_cnn_autocast_dtype(
    precision: Literal["none", "bf16"], device: T.device | str
) -> T.dtype | None:
    """Resolve the optional CNN autocast dtype for the selected device.

    Args:
        precision: CNN precision mode from run settings.
        device: Target execution device.

    Returns:
        ``torch.bfloat16`` when BF16 CNN autocast is enabled, otherwise None.

    Raises:
        ValueError: If BF16 CNN autocast is requested on a non-CUDA device.
        RuntimeError: If the selected CUDA device does not support BF16.
    """
    if precision == "none":
        return None

    resolved_device = T.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("BF16 CNN autocast requires a CUDA device")
    with T.cuda.device(resolved_device):
        if not T.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device {resolved_device} does not support BF16")
    return T.bfloat16
