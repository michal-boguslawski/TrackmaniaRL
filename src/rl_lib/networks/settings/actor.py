from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
)

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.layer import LinearLayerConfig


class ActorConfig(StrictModel):
    """Beta-distribution actor configuration.

    Attributes:
        hidden_layers: MLP hidden layers before output.
        output_init_gain: Gain for output layer weight init (small for Beta).
        output_init_bias: Bias for output layer (negative to start near 0).
        output_bias: Include bias in output layer.
        concentration_offset: Added to softplus output to ensure >= 1.
        softplus_beta: Beta parameter for softplus.
        softplus_threshold: Threshold parameter for softplus.
        deterministic_distribution_temperature: Temperature for deterministic eval.
    """

    hidden_layers: list[LinearLayerConfig] = Field(
        default_factory=lambda: [
            LinearLayerConfig(out_dim=256, activation="gelu"),
        ]
    )
    output_init_gain: PositiveFloat = 0.01
    output_init_bias: FiniteFloat = -2.0
    output_bias: bool = True
    concentration_offset: PositiveFloat = 1.0
    softplus_beta: PositiveFloat = 1.0
    softplus_threshold: PositiveFloat = 20.0
    deterministic_distribution_temperature: PositiveFloat = 1.0
