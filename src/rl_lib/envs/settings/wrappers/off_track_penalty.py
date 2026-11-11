from typing import Literal

from pydantic import Field, FiniteFloat

from rl_lib.config_base import StrictModel


class OffTrackPenaltyWrapperSettings(StrictModel):
    """Settings for penalizing distance beyond the track edge.

    Attributes:
        penalty_weight: Maximum magnitude of the per-step penalty.
    """

    name: Literal["off_track_penalty"] = Field(
        default="off_track_penalty", description="Select the off-track penalty wrapper."
    )
    penalty_weight: FiniteFloat = Field(
        default=0.05,
        ge=0.0,
        description="Maximum magnitude of the per-step penalty beyond the road edge.",
    )
