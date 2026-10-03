"""Convolutional encoder for image observations.

Processes NCHW image tensors through configurable conv layers, adaptive pooling,
and MLP head to produce fixed-dimensional feature vectors.
"""

import torch as T
import json
from torch import nn

from rl_lib.networks.config import CNNConfig
from rl_lib.networks.utils import init_layer, make_activation


class CNN(nn.Module):
    """Convolutional encoder with adaptive pooling and MLP projection.

    Architecture: Conv2d layers -> AdaptiveAvgPool2d -> Flatten -> MLP ->
    Linear(out_dim). Expects NCHW input normalized to [-1, 1].

    Attributes:
        observation_dim (int): Number of input channels (e.g., 1 for grayscale,
            3 for RGB, or stack_size for frame-stacked).
        cfg (CNNConfig): Configuration for conv layers, pooling, MLP, and init.
    """

    def __init__(
        self,
        observation_dim: int,
        config: CNNConfig,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.cfg = config

        layers: list[nn.Module] = []
        in_channels = observation_dim
        for layer_cfg in self.cfg.conv_layers:
            conv = nn.Conv2d(
                in_channels,
                layer_cfg.out_channels,
                kernel_size=layer_cfg.kernel_size,
                stride=layer_cfg.stride,
                padding=layer_cfg.padding,
                dilation=layer_cfg.dilation,
                groups=layer_cfg.groups,
                bias=layer_cfg.bias,
                padding_mode=layer_cfg.padding_mode,
            )
            layers.extend((
                init_layer(conv, layer_cfg.init_gain, layer_cfg.init_bias),
                make_activation(layer_cfg.activation),
            ))
            in_channels = layer_cfg.out_channels
        layers.append(nn.AdaptiveAvgPool2d((self.cfg.adaptive_pool_size, self.cfg.adaptive_pool_size)))
        layers.append(nn.Flatten())
        in_dim = in_channels * self.cfg.adaptive_pool_size**2
        for hidden_cfg in self.cfg.hidden_layers:
            linear = nn.Linear(in_dim, hidden_cfg.out_dim, bias=hidden_cfg.bias)
            layers.extend((
                init_layer(linear, hidden_cfg.init_gain, hidden_cfg.init_bias),
                make_activation(hidden_cfg.activation),
            ))
            in_dim = hidden_cfg.out_dim
        output = nn.Linear(in_dim, self.cfg.out_dim)
        layers.append(init_layer(output, self.cfg.output_init_gain, self.cfg.output_init_bias))
        self._network = nn.Sequential(*layers)

    @property
    def out_dim(self) -> int:
        """Output feature dimension after final linear layer."""
        return self.cfg.out_dim

    def layer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Group parameters per weighted layer, for layer-wise diagnostics.

        Layers are keyed by type and position in the stack, so the naming is
        unique and stays stable for a fixed configuration.

        Returns:
            Mapping from layer name (``conv_{i}`` / ``linear_{i}``) to that
            layer's parameters, weight first then bias.
        """
        groups: dict[str, list[nn.Parameter]] = {}
        indices = {"conv": 0, "linear": 0}
        for module in self._network:
            if isinstance(module, nn.Conv2d):
                prefix = "conv"
            elif isinstance(module, nn.Linear):
                prefix = "linear"
            else:
                continue
            name = f"{prefix}_{indices[prefix]}"
            indices[prefix] += 1
            groups[name] = list(module.parameters())
        return groups

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Encode image observation to feature vector.

        Args:
            x: Tensor of shape (batch, channel, height, width), dtype float32,
                normalized to [-1, 1] range.

        Returns:
            Tensor of shape (batch, out_dim).
        """
        return self._network(x)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        return f"{self.__class__.__name__}(observation_dim={self.observation_dim}, out_dim={self.out_dim}, config={self.cfg.model_dump()}, params={n_params:,})"

    def config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable hyperparameters."""
        config = {
            "observation_dim": self.observation_dim,
            "out_dim": self.out_dim,
            **{
                key: value if isinstance(value, (int, float, str, bool)) else json.dumps(value, sort_keys=True)
                for key, value in self.cfg.model_dump().items()
            },
            "n_params": sum(p.numel() for p in super().parameters()),
        }
        return config
