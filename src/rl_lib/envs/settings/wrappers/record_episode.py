from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class RecordEpisodeStatisticsWrapperSettings(StrictModel):
    """Settings for tracking episode rewards and lengths.

    Attributes:
        stats_buffer_length: Number of recent episode results to retain.
        stats_key: Info key containing episode statistics.
    """

    name: Literal["record_episode_stats"] = Field(
        default="record_episode_stats",
        description="Select the episode-statistics wrapper.",
    )
    stats_buffer_length: PositiveInt = Field(
        default=100, description="Number of recent episode results to retain."
    )
    stats_key: str = Field(
        default="episode",
        min_length=1,
        description="Info key used for episode statistics.",
    )
