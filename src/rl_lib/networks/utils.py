"""Network initialization utilities.

Provides orthogonal weight initialization and activation function factory
for consistent network construction across modules.
"""

from torch import nn
from rl_lib.networks.config import LinearLayerConfig


def init_layer(
    layer: nn.Module,
    gain: float | None = None,
    bias: float | None = None,
) -> nn.Module:
    """Initialize layer weights with orthogonal initialization.

    Uses LinearLayerConfig defaults if gain/bias not provided.

    Args:
        layer: nn.Module with weight and optional bias (Linear, Conv2d, etc.).
        gain: Orthogonal initialization gain (default: sqrt(2) for ReLU).
        bias: Bias initialization value (default: 0.0).

    Returns:
        The initialized layer (for chaining).
    """
    defaults = LinearLayerConfig(out_dim=1)
    gain = defaults.init_gain if gain is None else gain
    bias = defaults.init_bias if bias is None else bias
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


def make_activation(name: str) -> nn.Module:
    """Create activation module by name.

    Args:
        name: One of "relu", "gelu", "tanh", "elu", "identity".

    Returns:
        Instantiated activation module.

    Raises:
        KeyError: If name not in supported activations.
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "identity": nn.Identity,
    }
    return activations[name]()
