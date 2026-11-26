from dataclasses import dataclass

import torch as T
from torch.distributions import Distribution
from torch.optim import Optimizer


@dataclass
class UpdateContext:
    values: T.Tensor
    returns: T.Tensor
    dist: Distribution


@dataclass
class RolloutContext:
    action: T.Tensor
    returns: T.Tensor
    advantages: T.Tensor
    dones: T.Tensor
    values: T.Tensor


@dataclass
class OptimizationContext:
    optimizer: Optimizer
    clip_norm: T.Tensor
