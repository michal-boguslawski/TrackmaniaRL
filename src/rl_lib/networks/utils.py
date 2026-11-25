"""Network initialization utilities.

Provides orthogonal weight initialization and activation function factory
for consistent network construction across modules.
"""

from collections.abc import MutableMapping
from typing import Any

from torch import nn

from rl_lib.networks.settings.layer import LinearLayerSettings

DECAY_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)


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
    defaults = LinearLayerSettings(out_dim=1)
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


def update_state_dict(
    target: dict[str, nn.Parameter],
    source: dict[str, nn.Parameter],
    strict: bool = True,
) -> None:
    """Update target state dict with source state dict.

    Args:
        target: Target state dict to update.
        source: Source state dict to copy from.
        strict: If True, raise error if keys mismatch; else ignore missing keys.
    """
    for key, source_value in source.items():
        if key in target:
            target[key].data.copy_(source_value.data)
        elif strict:
            raise KeyError(f"Key {key} not found in target state dict.")


def update_weighted_state_dict(
    target: dict[str, nn.Parameter],
    source: dict[str, nn.Parameter],
    alpha: float,
) -> None:
    """Weighted update of target state dict with source state dict.

    Args:
        target: Target state dict to update.
        source: Source state dict to copy from.
        alpha: Weighting factor (0.0 = no update, 1.0 = full update).
    """
    for key, source_value in source.items():
        if key in target:
            target[key].data.mul_(1 - alpha).add_(source_value.data, alpha=alpha)


from torch import nn
from torch.nn.parameter import Parameter


def split_parameter_groups(
    module: nn.Module,
) -> dict[str, list[Parameter]]:
    groups = {
        "decay": [],
        "no_decay": [],
    }

    for submodule in module.modules():
        for name, param in submodule.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            if isinstance(submodule, DECAY_MODULES) and name == "weight":
                groups["decay"].append(param)
            else:
                groups["no_decay"].append(param)

    return groups


def flatten_dict(
    dict_: MutableMapping, parent_key: str = "", separator: str = "/"
) -> dict[str, str | int | float]:
    items = []
    for key, value in dict_.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        elif isinstance(value, list) and isinstance(value[0], MutableMapping | list):
            for i, item in enumerate(value):
                new_key += "_" + str(i)
                items.extend(flatten_dict(item, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
