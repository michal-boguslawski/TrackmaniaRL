from torch import nn
from rl_lib.networks.config import LinearLayerConfig


def init_layer(
    layer: nn.Module,
    gain: float | None = None,
    bias: float | None = None,
) -> nn.Module:
    defaults = LinearLayerConfig(out_dim=1)
    gain = defaults.init_gain if gain is None else gain
    bias = defaults.init_bias if bias is None else bias
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


def make_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "identity": nn.Identity,
    }
    return activations[name]()
