"""Buffer utilities for numpy/torch conversion."""

from numpy.typing import NDArray
import torch as T


def to_tensor_batch(minibatch: dict[str, NDArray], device: T.device) -> dict[str, T.Tensor]:
    """Convert numpy minibatch dict to torch tensors on device.

    Args:
        minibatch: Dict mapping keys to numpy arrays.
        device: Target torch device.

    Returns:
        Dict with same keys, values as torch tensors on device.
    """
    return {k: T.from_numpy(v).to(device) for k, v in minibatch.items()}
