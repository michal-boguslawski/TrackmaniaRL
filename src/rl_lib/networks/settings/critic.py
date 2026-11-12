from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
)

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.layer import LinearLayerSettings


class VCriticSettings(StrictModel):
    """Critic (value head) configuration.

    Attributes:
        hidden_layers: MLP hidden layers.
        output_init_gain: Gain for output layer weight init.
        output_init_bias: Bias for output layer init.
        output_bias: Include bias in output layer.
    """

    hidden_layers: list[LinearLayerSettings] = Field(
        default_factory=lambda: [
            LinearLayerSettings(out_dim=256, activation="gelu"),
        ]
    )
    output_init_gain: PositiveFloat = 1.0
    output_init_bias: FiniteFloat = 0.0
    output_bias: bool = True
