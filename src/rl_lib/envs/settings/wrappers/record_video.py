from collections.abc import Callable
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt

from rl_lib.envs.settings.wrappers.base import BaseWrapperSettings


class RecordVideoWrapperSettings(BaseWrapperSettings):
    """Settings for recording environment episodes to video.

    Attributes:
        video_folder: Output directory, inherited from environment settings if unset.
        episode_trigger: Whether to record all episodes or an interval.
        episode_interval: Interval used by the interval trigger.
        video_length: Maximum recorded episode length.
        video_name_prefix: Prefix used for video filenames.
        video_fps: Optional recorded video frame rate.
        video_disable_logger: Whether to disable Gymnasium's video logger.
    """

    name: Literal["record_video"] = Field(
        default="record_video", description="Select the video-recording wrapper."
    )
    video_folder: str = Field(
        description="Output directory; defaults to the environment video folder.",
    )
    episode_trigger: Callable[[int], bool] | None = Field(
        description="Record all episodes or episodes at an interval."
    )
    video_length: NonNegativeInt = Field(
        default=0,
        description="Maximum recorded episode length; zero records complete episodes.",
    )
    name_prefix: str = Field(
        default="rl-video", description="Prefix used for recorded video filenames."
    )
    fps: PositiveInt | None = Field(
        default=None, description="Optional playback frame rate override."
    )
    disable_logger: bool = Field(
        default=True, description="Disable Gymnasium's video recorder logger."
    )
