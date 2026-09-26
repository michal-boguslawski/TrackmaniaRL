# rl_lib/networks/config.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveFloat, PositiveInt


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConvLayerConfig(StrictConfig):
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


class LinearLayerConfig(StrictConfig):
    out_dim: PositiveInt
    activation: Literal["relu", "gelu", "tanh", "elu", "identity"] = "relu"
    init_gain: PositiveFloat = 2**0.5
    init_bias: FiniteFloat = 0.0
    bias: bool = True


class CNNConfig(StrictConfig):
    conv_layers: list[ConvLayerConfig] = Field(default_factory=lambda: [
        ConvLayerConfig(out_channels=32, kernel_size=8, stride=4),
        ConvLayerConfig(out_channels=64, kernel_size=4, stride=2),
        ConvLayerConfig(out_channels=128, kernel_size=3),
        ConvLayerConfig(out_channels=128, kernel_size=3, stride=2),
    ])
    adaptive_pool_size: PositiveInt = 3
    hidden_layers: list[LinearLayerConfig] = Field(default_factory=lambda: [
        LinearLayerConfig(out_dim=256),
    ])
    out_dim: PositiveInt = 256
    output_init_gain: PositiveFloat = 2**0.5
    output_init_bias: FiniteFloat = 0.0


class TemporalConfig(StrictConfig):
    out_dim: PositiveInt = 256
    normalize_input: bool = True
    normalize_output: bool = True
    bias: bool = True
    norm_eps: PositiveFloat = 1e-5
    norm_affine: bool = True


class ActorConfig(StrictConfig):
    hidden_layers: list[LinearLayerConfig] = Field(default_factory=lambda: [
        LinearLayerConfig(out_dim=256, activation="gelu"),
    ])
    output_init_gain: PositiveFloat = 0.01
    output_init_bias: FiniteFloat = -2.0
    output_bias: bool = True
    action_low: list[FiniteFloat] = Field(default_factory=lambda: [-1.0, 0.0, 0.0])
    action_high: list[FiniteFloat] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    concentration_offset: PositiveFloat = 1.0
    softplus_beta: PositiveFloat = 1.0
    softplus_threshold: PositiveFloat = 20.0
    default_temperature: PositiveFloat = 1.0
    deterministic_distribution_temperature: PositiveFloat = 1.0


class CriticConfig(StrictConfig):
    hidden_layers: list[LinearLayerConfig] = Field(default_factory=lambda: [
        LinearLayerConfig(out_dim=256, activation="gelu"),
    ])
    output_init_gain: PositiveFloat = 1.0
    output_init_bias: FiniteFloat = 0.0
    output_bias: bool = True
    huber_delta: PositiveFloat = 1.0


class NetworkConfig(StrictConfig):
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
