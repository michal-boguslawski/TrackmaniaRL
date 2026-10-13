"""Typed settings for supported learning-rate scheduler variants."""

from typing import Annotated, Literal, Union

from pydantic import Field

from rl_lib.training.settings.base import StrictModel


class CosineSchedulerSettings(StrictModel):
    """Cosine annealing scheduler parameters.

    Attributes:
        name: Discriminator selecting cosine annealing.
        eta_min: Minimum learning rate.
        extra_iters: Additional iterations beyond the computed schedule length.
    """

    name: Literal["cosine"] = Field(
        default="cosine", description="Scheduler type discriminator."
    )
    eta_min: float = Field(
        default=1e-8,
        ge=0,
        description="Minimum learning rate for cosine annealing.",
    )
    extra_iters: int = Field(
        default=1,
        ge=0,
        description="Iterations added to the computed schedule length.",
    )


class LinearSchedulerSettings(StrictModel):
    """Linear learning-rate scheduler parameters.

    Attributes:
        name: Discriminator selecting linear decay.
        min_lr: Learning rate at the end of the schedule.
        start_factor: Initial learning-rate multiplier.
        extra_iters: Additional iterations beyond the computed schedule length.
    """

    name: Literal["linear"] = Field(
        default="linear", description="Scheduler type discriminator."
    )
    min_lr: float = Field(
        default=1e-8,
        ge=0,
        description="Minimum learning rate at the end of the schedule.",
    )
    start_factor: float = Field(
        default=1.0, gt=0, le=1, description="Initial rate multiplier."
    )
    extra_iters: int = Field(
        default=1,
        ge=0,
        description="Iterations added to the computed schedule length.",
    )


class NoSchedulerSettings(StrictModel):
    """Disabled learning-rate scheduler configuration.

    Attributes:
        name: Discriminator selecting no scheduler.
    """

    name: Literal["none"] = Field(
        default="none", description="Scheduler type discriminator."
    )


class ExponentialSchedulerSettings(StrictModel):
    """Exponential learning-rate scheduler parameters.

    Attributes:
        name: Discriminator selecting exponential decay.
        gamma: Multiplicative learning-rate decay per update.
    """

    name: Literal["exponential"] = Field(
        default="exponential", description="Scheduler type discriminator."
    )
    gamma: float = Field(
        default=0.99,
        gt=0,
        le=1,
        description="Multiplicative learning-rate decay per update.",
    )


SchedulerSettings = Annotated[
    Union[
        CosineSchedulerSettings,
        LinearSchedulerSettings,
        ExponentialSchedulerSettings,
        NoSchedulerSettings,
    ],
    Field(discriminator="name"),
]
