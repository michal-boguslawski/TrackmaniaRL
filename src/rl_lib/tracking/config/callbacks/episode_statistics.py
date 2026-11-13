"""Configuration settings for episode-statistics logging."""

from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class EpisodeStatisticsCallbackSettings(StrictModel):
    """Configure buffering and aggregation for completed-episode statistics."""

    name: Literal["episode_statistics"] = Field(
        default="episode_statistics",
        description="Callback registry name.",
    )
    mode: Literal["step", "mean"] = Field(
        default="mean",
        description=(
            "Log each buffered step summary separately, or log episode-weighted "
            "means for the buffer."
        ),
    )
    stats_key: str = Field(
        default="episode",
        min_length=1,
        description="Info-dict key populated by RecordEpisodeStatistics.",
    )
    buffer_length: PositiveInt = Field(
        default=100,
        description=(
            "Number of environment-step summaries to buffer before automatic "
            "logging. Partial buffers can be explicitly flushed."
        ),
    )
