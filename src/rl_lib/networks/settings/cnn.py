from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
    PositiveInt,
)

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.layer import ConvLayerConfig, LinearLayerConfig


class CNNConfig(StrictModel):
    """Convolutional encoder configuration.

    Attributes:
        conv_layers: List of Conv2d layer configs.
        adaptive_pool_size: Output size of AdaptiveAvgPool2d (H=W).
        hidden_layers: MLP layers after flattening.
        out_dim: Final output dimension.
        output_init_gain: Gain for final Linear layer weight init.
        output_init_bias: Bias for final Linear layer init.
    """

    conv_layers: list[ConvLayerConfig] = Field(
        default_factory=lambda: [
            ConvLayerConfig(out_channels=32, kernel_size=8, stride=4),
            ConvLayerConfig(out_channels=64, kernel_size=4, stride=2),
            ConvLayerConfig(out_channels=128, kernel_size=3),
            ConvLayerConfig(out_channels=128, kernel_size=3, stride=2),
        ]
    )
    adaptive_pool_size: PositiveInt = 3
    hidden_layers: list[LinearLayerConfig] = Field(
        default_factory=lambda: [
            LinearLayerConfig(out_dim=256),
        ]
    )
    out_dim: PositiveInt = 256
    output_init_gain: PositiveFloat = 2**0.5
    output_init_bias: FiniteFloat = 0.0
