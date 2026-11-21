from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class RecordVideoCallbackSettings(StrictModel):
    """Configure buffering and aggregation for completed-episode statistics."""

    name: Literal["record_video"] = Field(
        default="record_video",
        description="Callback registry name.",
    )
    video_folder: str = Field(
        description="Output directory; defaults to the environment video folder.",
    )
    interval: PositiveInt = Field(
        default=100_000,
    )
