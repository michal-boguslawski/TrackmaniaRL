"""Network configuration schemas (Pydantic models).

Defines the structure for CNN, temporal encoder, actor, and critic configs.
All configs use StrictConfig (extra="forbid") to catch typos in YAML.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveFloat, PositiveInt


class StrictConfig(BaseModel):
    """Base config with extra='forbid' to prevent silent typos in YAML."""
    model_config = ConfigDict(extra="forbid")


class ConvLayerConfig(StrictConfig):
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


class LinearLayerConfig(StrictConfig):
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


class CNNConfig(StrictConfig):
    """Convolutional encoder configuration.

    Attributes:
        conv_layers: List of Conv2d layer configs.
        adaptive_pool_size: Output size of AdaptiveAvgPool2d (H=W).
        hidden_layers: MLP layers after flattening.
        out_dim: Final output dimension.
        output_init_gain: Gain for final Linear layer weight init.
        output_init_bias: Bias for final Linear layer init.
    """
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
    """Temporal encoder (TemporalCNN1D) configuration.

    Attributes:
        out_dim: Output feature dimension.
        normalize_input: Apply LayerNorm to input sequence.
        normalize_output: Apply LayerNorm to output.
        bias: Include bias in Conv1d.
        norm_eps: LayerNorm epsilon.
        norm_affine: LayerNorm learnable affine parameters.
    """
    out_dim: PositiveInt = 256
    normalize_input: bool = True
    normalize_output: bool = True
    bias: bool = True
    norm_eps: PositiveFloat = 1e-5
    norm_affine: bool = True


class ActorConfig(StrictConfig):
    """Beta-distribution actor configuration.

    Attributes:
        hidden_layers: MLP hidden layers before output.
        output_init_gain: Gain for output layer weight init (small for Beta).
        output_init_bias: Bias for output layer (negative to start near 0).
        output_bias: Include bias in output layer.
        action_low: Lower bounds per action dimension (e.g., [-1, 0, 0]).
        action_high: Upper bounds per action dimension (e.g., [1, 1, 1]).
        concentration_offset: Added to softplus output to ensure >= 1.
        softplus_beta: Beta parameter for softplus.
        softplus_threshold: Threshold parameter for softplus.
        default_temperature: Default sampling temperature.
        deterministic_distribution_temperature: Temperature for deterministic eval.
    """
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
    """Critic (value head) configuration.

    Attributes:
        hidden_layers: MLP hidden layers.
        output_init_gain: Gain for output layer weight init.
        output_init_bias: Bias for output layer init.
        output_bias: Include bias in output layer.
        huber_delta: Delta for Huber loss used in critic training.
    """
    hidden_layers: list[LinearLayerConfig] = Field(default_factory=lambda: [
        LinearLayerConfig(out_dim=256, activation="gelu"),
    ])
    output_init_gain: PositiveFloat = 1.0
    output_init_bias: FiniteFloat = 0.0
    output_bias: bool = True
    huber_delta: PositiveFloat = 1.0


class NetworkConfig(StrictConfig):
    """Complete network configuration composing all sub-configs.

    Attributes:
        cnn: Convolutional encoder config.
        temporal: Temporal encoder config.
        actor: Actor config.
        critic: Critic config.
    """
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
