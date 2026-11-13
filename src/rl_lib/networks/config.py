"""Network configuration schemas (Pydantic models).

Defines the structure for CNN, temporal encoder, actor, and critic configs.
All configs use StrictConfig (extra="forbid") to catch typos in YAML.
"""

from typing import Literal

from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
    PositiveInt,
)

from rl_lib.config_base import StrictModel


class NetworkConfig(StrictConfig):
    """Complete network configuration composing all sub-configs.

    Attributes:
        encoder: # TODO
        actor: Actor config.
        critic: Critic config.
    """

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)
    critic: VCriticConfig = Field(default_factory=VCriticConfig)


class SACActorConfig(StrictConfig):
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)


class SACTwinCriticConfig(StrictConfig):
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    critic: VCriticConfig = Field(default_factory=VCriticConfig)


class SACNetworkConfig(StrictConfig):
    actor: SACActorConfig = Field(default_factory=SACActorConfig)
    twin_critic: SACTwinCriticConfig = Field(default_factory=SACTwinCriticConfig)
