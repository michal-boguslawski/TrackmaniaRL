from typing import Any

from src.rl_lib.tracking.base import MetricsLogger
from src.rl_lib.training.callbacks.base import CollectorCallback

class RecordStatisticLoggerCallback(CollectorCallback):
    def __init__(self, logger: MetricsLogger):
        self._logger = logger

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, info: dict[str, Any], *args, **kwargs):
        if "episode" in info:
            dones = info["_episode"]
            returns = info["episode"]["r"][dones]
            lengths = info["episode"]["l"][dones]
            metrics = {
                "episode/returns": returns.mean(),
                "episode/lengths": lengths.mean(),
            }
            self._logger.log_metrics(metrics, step=step)

    def on_rollout_end(self, *args, **kwargs):
        pass