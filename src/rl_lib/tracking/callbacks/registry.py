from rl_lib.tracking.callbacks.base import Callback
from rl_lib.tracking.callbacks.episode_statistics import EpisodeStatisticLoggerCallback
from rl_lib.tracking.callbacks.evaluate import EvaluationCallback
from rl_lib.tracking.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.tracking.callbacks.model_checkpoint import ModelCheckpointCallback
from rl_lib.tracking.callbacks.record_video import RecordVideoCallback

CALLBACK_REGISTRY: dict[str, type[Callback]] = {
    "metrics_logger": MetricsLoggingCallback,
    "episode_statistics": EpisodeStatisticLoggerCallback,
    "record_video": RecordVideoCallback,
    "evaluation": EvaluationCallback,
    "model_checkpoint": ModelCheckpointCallback,
}
