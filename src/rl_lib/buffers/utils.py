from numpy.typing import NDArray
import torch as T


def to_tensor_batch(minibatch: dict[str, NDArray], device: T.device) -> dict[str, T.Tensor]:
    return {k: T.from_numpy(v).to(device) for k, v in minibatch.items()}
