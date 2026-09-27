"""Device resolution utility.

Provides a single function to resolve the torch device from run configuration,
with automatic CUDA detection and validation.
"""

from __future__ import annotations

import torch as T

from rl_lib.run_config import RunConfig


def resolve_device(config: RunConfig) -> T.device:
    """Resolve the configured torch device, validating CUDA availability.

    Args:
        config: RunConfig containing run.device setting ("auto", "cuda", "cpu",
            or specific device like "cuda:0").

    Returns:
        torch.device ready for model/data placement.

    Raises:
        RuntimeError: If explicit CUDA device requested but CUDA unavailable.
    """
    if config.run.device == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(config.run.device)
    if device.type == "cuda" and not T.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    return device
