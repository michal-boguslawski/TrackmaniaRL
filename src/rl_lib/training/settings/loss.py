"""Algorithm-specific loss settings."""

from typing import Literal

from pydantic import Field, FiniteFloat, PositiveFloat, model_validator

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


class PPOActorLossSettings(StrictModel):
    advantage_normalization_strategy: Literal["batch", "global"] | None = Field(
        default=None,
        description="Normalize advantages per minibatch, globally, or not at all.",
    )
    log_ratio_min: FiniteFloat = Field(
        default=-5.0, description="Lower bound applied to policy log ratios."
    )
    log_ratio_max: FiniteFloat = Field(
        default=5.0, description="Upper bound applied to policy log ratios."
    )
    advantage_epsilon: float = Field(
        default=1e-4,
        gt=0,
        description="Small constant used when normalizing advantages.",
    )
    ppo_epsilon: float = Field(
        default=0.2, gt=0, lt=1, description="PPO policy clipping range."
    )
    mean_reg_coef: float = Field(
        default=0.01, ge=0, description="Policy mean regularization coefficient."
    )

    @model_validator(mode="after")
    def validate_log_ratio_range(self):
        """Require an increasing interval for policy log-ratio clamping."""
        if self.log_ratio_min >= self.log_ratio_max:
            raise ValueError(
                "trainer.stability.log_ratio_min must be lower than log_ratio_max"
            )
        return self


class PPOCriticLossSettings(StrictModel):
    """PPO critic loss settings."""

    value_clip_epsilon: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Optional value clipping range; defaults to ppo_epsilon.",
    )
    critic_beta: float = Field(default=0.5, ge=0, description="Value loss coefficient.")
    huber_delta: PositiveFloat = Field(default=1.0, description="Huber loss delta.")


class PPOLossSettings(StrictModel):
    """PPO actor, critic, entropy, and advantage objective settings."""

    name: Literal["ppo"] = Field(
        default="ppo",
        description="Loss family discriminator for algorithm-specific settings.",
    )
    actor_loss_cfg: PPOActorLossSettings = Field(
        default_factory=PPOActorLossSettings,
    )
    critic_loss_cfg: PPOCriticLossSettings = Field(
        default_factory=PPOCriticLossSettings,
    )
    entropy: EntropySettings = Field(
        default_factory=EntropySettings,
        description="Entropy bonus coefficient and decay schedule.",
    )
    logs_diagnostics: bool = Field(
        default=True,
    )
