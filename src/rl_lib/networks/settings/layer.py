from typing import Literal

from pydantic import (
    FiniteFloat,
    PositiveFloat,
    PositiveInt,
)

from rl_lib.config_base import StrictModel


class ConvLayerSettings(StrictModel):
    """Single Conv2d layer configuration.

    Attributes:
        out_channels: Number of output channels.
        kernel_size: Square kernel size.
        stride: Convolution stride.
        padding: Padding (can be negative for valid padding with dilation).
        dilation: Dilation rate.
        groups: Number of groups for grouped convolution.
        bias: Whether to include bias term.
        padding_mode: Padding mode for Conv2d.
        activation: Activation function after convolution.
        init_gain: Weight initialization gain (sqrt(2) for ReLU, 1 for linear).
        init_bias: Bias initialization value.
    """

    out_channels: PositiveInt
    kernel_size: PositiveInt
    stride: PositiveInt = 1
    padding: int = 0
    dilation: PositiveInt = 1
    groups: PositiveInt = 1
    bias: bool = True
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
    activation: Literal["relu", "gelu", "tanh", "elu", "identity"] = "relu"
    init_gain: PositiveFloat = 2**0.5
    init_bias: FiniteFloat = 0.0


class LinearLayerSettings(StrictModel):
    """Single Linear layer configuration.

    Attributes:
        out_dim: Output dimension.
        activation: Activation function after linear layer.
        init_gain: Weight initialization gain.
        init_bias: Bias initialization value.
        bias: Whether to include bias term.
    """

    out_dim: PositiveInt
    activation: Literal["relu", "gelu", "tanh", "elu", "identity"] = "relu"
    init_gain: PositiveFloat = 2**0.5
    init_bias: FiniteFloat = 0.0
    bias: bool = True
