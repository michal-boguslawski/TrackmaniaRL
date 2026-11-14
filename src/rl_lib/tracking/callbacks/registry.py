from rl_lib.tracking.callbacks.base import Callback
from rl_lib.tracking.callbacks.episode_statistics import EpisodeStatisticLoggerCallback
from rl_lib.tracking.callbacks.metrics_logger import MetricsLoggingCallback

CALLBACK_REGISTRY: dict[str, type[Callback]] = {
    "metrics_logger": MetricsLoggingCallback,
    "episode_statistics": EpisodeStatisticLoggerCallback,
}
