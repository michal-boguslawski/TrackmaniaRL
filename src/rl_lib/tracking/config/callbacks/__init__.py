from typing import Annotated

from pydantic import Field

from rl_lib.tracking.config.callbacks.episode_statistics import (
    EpisodeStatisticsCallbackSettings,
)
from rl_lib.tracking.config.callbacks.metrics_logger import MetricsCallbackSettings

CallbackSettings = Annotated[
    EpisodeStatisticsCallbackSettings | MetricsCallbackSettings,
    Field(discriminator="name"),
]
