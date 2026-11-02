import torch as T


def episode_mask(dones: T.Tensor) -> T.Tensor:
    """
    # TODO
    """

    mask = dones.logical_not() & (dones.flip(1).cumsum(dim=1).flip(1) > 0)
    return mask
