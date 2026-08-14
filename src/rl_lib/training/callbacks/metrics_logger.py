import numpy as np
from typing import Literal, Any

from src.rl_lib.tracking.base import MetricsLogger
from src.rl_lib.training.callbacks.base import TrainingCallback


class MetricsLoggingCallback(TrainingCallback):
    def __init__(self, logger: MetricsLogger, granularity: Literal["minibatch", "epoch", "batch"] = "epoch"):
        self._logger = logger
        self.granularity = granularity
        self._epoch_buffer: list[dict[str, float]] = []
        self._step = 0

    def on_start(self, step: int, metrics: dict[str, float], *args, **kwargs):
        self._logger.log_metrics(metrics, step=step)
        self._epoch_buffer.clear()

    def on_minibatch(self, metrics: dict[str, float], **kwargs):
        if self.granularity == "minibatch":
            self._step += 1
            self._logger.log_metrics(metrics, step=self._step)
        else:
            self._epoch_buffer.append(metrics)

    def on_epoch(self, *args, **kwargs):
        if self.granularity == "epoch" and self._epoch_buffer:
            keys = self._epoch_buffer[0].keys()
            aggregated = {k: float(np.mean([m[k] for m in self._epoch_buffer])) for k in keys}
            self._step += 1
            self._logger.log_metrics(aggregated, step=self._step)
            self._epoch_buffer.clear()

    def on_end(self, *args, **kwargs):
        if self.granularity == "batch" and self._epoch_buffer:
            keys = self._epoch_buffer[0].keys()
            aggregated = {k: float(np.mean([m[k] for m in self._epoch_buffer])) for k in keys}
            self._step += 1
            self._logger.log_metrics(aggregated, step=self._step)
            self._epoch_buffer.clear()
