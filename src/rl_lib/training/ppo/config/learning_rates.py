"""PPO-specific learning rates for optimizer parameter groups."""

from pydantic import Field

from rl_lib.config_base import StrictModel


class PPOLearningRateSettings(StrictModel):
    """Learning rates for the PPO network components.

    Component rates omitted from the configuration fall back to the actor rate
    after validation, so every parameter group receives an explicit rate.

    Attributes:
        cnn_lr: CNN feature extractor rate; defaults to ``actor_lr``.
        sequence_encoder_lr: Temporal encoder rate; defaults to ``actor_lr``.
        actor_lr: Policy head rate.
        critic_lr: Value head rate.
    """

    cnn_lr: float | None = Field(
        default=None, gt=0, description="CNN rate; defaults to actor_lr."
    )
    sequence_encoder_lr: float | None = Field(
        default=None,
        gt=0,
        description="Sequence encoder rate; defaults to actor_lr.",
    )
    actor_lr: float = Field(default=1e-4, gt=0, description="Actor learning rate.")
    critic_lr: float = Field(default=1e-4, gt=0, description="Critic learning rate.")

    def __post_init__(self) -> None:
        """Fill optional component rates from the actor rate."""
        if self.cnn_lr is None:
            self.cnn_lr = self.actor_lr
        if self.sequence_encoder_lr is None:
            self.sequence_encoder_lr = self.actor_lr
