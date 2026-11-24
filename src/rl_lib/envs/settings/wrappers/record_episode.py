from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.envs.settings.wrappers.base import BaseWrapperSettings


class RecordEpisodeStatisticsWrapperSettings(BaseWrapperSettings):
    """Settings for tracking episode rewards and lengths.

    Attributes:
        stats_buffer_length: Number of recent episode results to retain.
        stats_key: Info key containing episode statistics.
    """

    name: Literal["record_episode_stats"] = Field(
        default="record_episode_stats",
        description="Select the episode-statistics wrapper.",
    )
    buffer_length: PositiveInt = Field(
        default=100, description="Number of recent episode results to retain."
    )
    stats_key: str = Field(
        default="episode",
        min_length=1,
        description="Info key used for episode statistics.",
    )
