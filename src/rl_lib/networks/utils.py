import numpy as np
from torch import nn


def init_layer(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer
