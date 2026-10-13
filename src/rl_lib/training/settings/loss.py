"""Algorithm-specific loss settings."""

from typing import Literal

from pydantic import Field

from rl_lib.training.settings.base import StrictModel


class EntropySettings(StrictModel):
    """Initial entropy bonus and its per-update decay schedule."""

    coefficient: float = Field(
        default=0.001, ge=0, description="Initial entropy bonus coefficient."
    )
    decay: float = Field(
        default=0.995,
        gt=0,
        le=1,
        description="Multiplicative entropy coefficient decay per update.",
    )
    min_coefficient: float = Field(
        default=1e-4,
        ge=0,
        description="Lower bound for the decayed entropy coefficient.",
    )


class PPOLossSettings(StrictModel):
    """PPO actor, critic, entropy, and advantage objective settings."""

    name: Literal["ppo"] = Field(
        default="ppo",
        description="Loss family discriminator for algorithm-specific settings.",
    )
    ppo_epsilon: float = Field(
        default=0.2, gt=0, lt=1, description="PPO policy clipping range."
    )
    value_clip_epsilon: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Optional value clipping range; defaults to ppo_epsilon.",
    )
    critic_beta: float = Field(
        default=0.5, ge=0, description="Value loss coefficient."
    )
    entropy: EntropySettings = Field(
        default_factory=EntropySettings,
        description="Entropy bonus coefficient and decay schedule.",
    )
    mean_reg_coef: float = Field(
        default=0.01, ge=0, description="Policy mean regularization coefficient."
    )
    advantage_normalization_strategy: Literal["batch", "global"] | None = Field(
        default=None,
        description="Normalize advantages per minibatch, globally, or not at all.",
    )
