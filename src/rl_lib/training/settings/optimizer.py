"""Optimizer settings and algorithm-specific parameter-group rates."""

from typing import Literal

from pydantic import Field

from rl_lib.config_base import StrictModel
from rl_lib.training.ppo.config.learning_rates import PPOLearningRateSettings


class OptimizerSettings(StrictModel):
    """Optimizer choice, PPO learning rates, and Adam-family options.

    Attributes:
        name: Adam or AdamW implementation.
        learning_rates: PPO learning rates for the network parameter groups.
        weight_decay: Decay for regularized parameter groups.
        no_decay_weight_decay: Decay for bias and normalization groups.
        eps: Adam numerical stability constant.
        beta1: First-moment coefficient.
        beta2: Second-moment coefficient.
        amsgrad: Whether to enable AMSGrad.
        fused: Whether to use fused CUDA kernels when available.
    """

    name: Literal["AdamW", "Adam"] = Field(
        default="AdamW", description="Optimizer implementation."
    )
    learning_rates: PPOLearningRateSettings = Field(
        default_factory=PPOLearningRateSettings,
        description="Learning rates for the PPO network parameter groups.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="Weight decay for decay parameter groups.",
    )
    no_decay_weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="Weight decay for bias and normalization groups.",
    )
    eps: float = Field(
        default=1e-5, gt=0, description="Adam numerical stability epsilon."
    )
    beta1: float = Field(
        default=0.9, ge=0, lt=1, description="Adam first-moment coefficient."
    )
    beta2: float = Field(
        default=0.999, ge=0, lt=1, description="Adam second-moment coefficient."
    )
    amsgrad: bool = Field(default=False, description="Enable the Adam AMSGrad variant.")
    fused: bool = Field(
        default=True,
        description="Use fused Adam/AdamW kernels on CUDA; ignored on CPU.",
    )
