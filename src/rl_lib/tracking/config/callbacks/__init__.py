from typing import Annotated

from pydantic import Field

from rl_lib.tracking.config.callbacks.episode_statistics import (
    EpisodeStatisticsCallbackSettings,
)
from rl_lib.tracking.config.callbacks.evaluation import EvaluationCallbackSettings
from rl_lib.tracking.config.callbacks.metrics_logger import MetricsCallbackSettings
from rl_lib.tracking.config.callbacks.model_checkpoint import (
    ModelCheckpointCallbackSettings,
)
from rl_lib.tracking.config.callbacks.record_video import RecordVideoCallbackSettings

CallbackSettings = Annotated[
    EpisodeStatisticsCallbackSettings
    | MetricsCallbackSettings
    | RecordVideoCallbackSettings
    | ModelCheckpointCallbackSettings
    | EvaluationCallbackSettings,
    Field(discriminator="name"),
]
