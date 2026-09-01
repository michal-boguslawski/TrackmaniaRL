import numpy as np
from typing import Literal, Any

from src.rl_lib.tracking.base import MetricsLogger
from src.rl_lib.training.callbacks.base import TrainingCallback


class MetricsLoggingCallback(TrainingCallback):
    def __init__(self, logger: MetricsLogger, granularity: Literal["minibatch", "epoch", "batch"] = "epoch"):
        self._logger = logger
        self.granularity = granularity
        self._epoch_buffer: list[dict[str, float]] = []
        self._log_buffer: list[tuple[dict[str, float], int]] = []
        self._step = 0

    def _add_to_buffer(self, metrics: dict[str, float], step: int):
        self._log_buffer.append((metrics, step))

    def flush(self):
        merged: dict[int, dict[str, float]] = {}
        for metrics, step in self._log_buffer:
            merged.setdefault(step, {}).update(metrics)

        for step, metrics in merged.items():
            self._logger.log_metrics(metrics, step=step)

        self._log_buffer.clear()

    def on_start(self, step: int, metrics: dict[str, float], *args, **kwargs):
        self._add_to_buffer(metrics, step)
        self._epoch_buffer.clear()

    def on_minibatch(self, metrics: dict[str, float], **kwargs):
        if self.granularity == "minibatch":
            self._step += 1
            self._add_to_buffer(metrics, self._step)
        else:
            self._epoch_buffer.append(metrics)

    def on_epoch(self, *args, **kwargs):
        if self.granularity == "epoch" and self._epoch_buffer:
            keys = self._epoch_buffer[0].keys()
            aggregated = {k: float(np.mean([m[k] for m in self._epoch_buffer])) for k in keys}
            self._step += 1
            self._add_to_buffer(aggregated, self._step)
            self._epoch_buffer.clear()

    def on_end(self, step: int, metrics: dict[str, float] | None, *args, **kwargs):
        if metrics:
            self._add_to_buffer(metrics, step)

        if self.granularity == "batch" and self._epoch_buffer:
            keys = self._epoch_buffer[0].keys()
            aggregated = {k: float(np.mean([m[k] for m in self._epoch_buffer])) for k in keys}
            self._step += 1
            self._add_to_buffer(aggregated, step)
            self._epoch_buffer.clear()

        self.flush()
