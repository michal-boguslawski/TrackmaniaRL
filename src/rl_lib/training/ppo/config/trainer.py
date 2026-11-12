"""Top-level trainer settings and compatibility migration."""

from typing import Literal

from pydantic import Field

from rl_lib.config_base import StrictModel
from rl_lib.training.ppo.config.loss import PPOLossConfig
from rl_lib.training.settings.optimizer import OptimizerSettings
from rl_lib.training.settings.scheduler import (
    CosineSchedulerSettings,
    SchedulerSettings,
)


class KLControlSettings(StrictModel):
    """KL-based early stopping controls for PPO updates."""

    target_kl: float = Field(
        default=0.03, gt=0, description="Target approximate KL per update."
    )
    warmup_steps: int = Field(
        default=20_000, ge=0, description="Updates before KL stopping is enabled."
    )
    stop_multiplier: float = Field(
        default=3.0,
        gt=0,
        description="Multiplier applied to target_kl for hard stopping.",
    )
    hard_stop: bool = Field(
        default=True, description="Stop PPO updates when the KL threshold is exceeded."
    )


class GradientSettings(StrictModel):
    """Gradient clipping limits for PPO network components."""

    backbone_max_grad_norm: float = Field(
        default=0.5,
        gt=0,
        description="Maximum gradient norm for the CNN and sequence encoder.",
    )
    actor_max_grad_norm: float = Field(
        default=0.5, gt=0, description="Maximum gradient norm for the actor."
    )
    critic_max_grad_norm: float = Field(
        default=0.5, gt=0, description="Maximum gradient norm for the critic."
    )


# class StabilitySettings(StrictModel):
#     """Numerical guards used by PPO loss and metric calculations."""

#     advantage_epsilon: float = Field(default=1e-4, gt=0, description="Small constant used when normalizing advantages.")
#     explained_variance_epsilon: float = Field(default=1e-8, gt=0, description="Small constant preventing a zero variance denominator.")
#     log_ratio_min: FiniteFloat = Field(default=-5.0, description="Lower bound applied to policy log ratios.")
#     log_ratio_max: FiniteFloat = Field(default=5.0, description="Upper bound applied to policy log ratios.")
#     skip_nonfinite_updates: bool = Field(default=True, description="Skip optimizer updates with non-finite losses or gradients.")
#     tanh_saturation_threshold: float = Field(default=3.5, gt=0, description="Absolute action-mean threshold used for saturation diagnostics.")

#     @model_validator(mode="after")
#     def validate_log_ratio_range(self):
#         """Require an increasing interval for policy log-ratio clamping."""
#         if self.log_ratio_min >= self.log_ratio_max:
#             raise ValueError("trainer.stability.log_ratio_min must be lower than log_ratio_max")
#         return self


class PPOTrainerSettings(StrictModel):
    """Trainer identity and grouped algorithm-specific optimization settings.

    Loss settings are algorithm-specific, so a future SAC trainer can provide a
    different loss configuration without adding SAC parameters to the PPO loss
    model. Optimizer and scheduler settings remain independently typed.

    Attributes:
        name: Trainer or algorithm identity.
        loss: PPO objective and entropy settings.
        kl: KL-based update stopping settings.
        gradients: Component-wise gradient clipping settings.
        stability: Numerical guards for losses and diagnostics.
        optimizer: Optimizer and parameter-group settings.
        scheduler: Learning-rate scheduler variant and settings.
        minibatch_indexing_mode: Whether minibatch indices are IID or sequential.
    """

    loss: PPOLossConfig = Field(
        default_factory=PPOLossConfig,
    )
    kl: KLControlSettings = Field(
        default_factory=KLControlSettings,
        description="KL-based stopping behavior.",
    )
    # gradients: GradientSettings = Field(
    #     default_factory=GradientSettings,
    #     description="Gradient clipping limits.",
    # )
    # stability: StabilitySettings = Field(
    #     default_factory=StabilitySettings,
    #     description="Numerical stability and non-finite update behavior.",
    # )
    optimizer: OptimizerSettings = Field(
        default_factory=OptimizerSettings,
        description="Optimizer implementation and parameter-group settings.",
    )
    scheduler: SchedulerSettings = Field(
        default_factory=CosineSchedulerSettings,
        description="Learning-rate scheduler variant and its options.",
    )
    minibatch_indexing_mode: Literal["iid", "sequential"] = Field(
        default="iid",
        description="Minibatch sample ordering strategy.",
    )
