from __future__ import annotations

import torch as T

from rl_lib.run_config import RunConfig


def resolve_device(config: RunConfig) -> T.device:
    """Resolve the configured torch device, validating CUDA availability."""
    if config.run.device == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(config.run.device)
    if device.type == "cuda" and not T.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    return device
