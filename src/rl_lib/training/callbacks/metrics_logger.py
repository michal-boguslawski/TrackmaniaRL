"""Training callback: logs metrics to backends at configurable granularity.

Buffers metrics per minibatch/epoch/batch and flushes to MetricsLogger
backends (console, MLflow). Supports three granularity modes:
- minibatch: log every minibatch step
- epoch: average and log once per epoch
- batch: average and log once per full batch (after all epochs)
"""

import numpy as np

from rl_lib.run_config import MetricsCallbackSettings
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import Callback


class MetricsLoggingCallback(Callback):
    """Log training metrics at specified granularity.

    Attributes:
        _logger: MetricsLogger backend (console, MLflow, etc.).
        cfg: MetricsCallbackSettings with granularity setting.
        _epoch_buffer: Accumulates metrics within an epoch.
        _log_buffer: Accumulates (metrics, step) pairs for flushing.
        _step: Internal step counter for minibatch/epoch logging.
    """

    def __init__(self, logger: MetricsLogger, config: MetricsCallbackSettings):
        self._logger = logger
        self.cfg = config
        self._epoch_buffer: list[dict[str, float]] = []
        self._log_buffer: list[tuple[dict[str, float], int]] = []
        self._step = 0

    def _add_to_buffer(self, metrics: dict[str, float], step: int):
        """Queue metrics for later flush."""
        self._log_buffer.append((metrics, step))

    def flush(self):
        """Write all buffered metrics to logger, merging by step."""
        merged: dict[int, dict[str, float]] = {}
        for metrics, step in self._log_buffer:
            merged.setdefault(step, {}).update(metrics)

        for step, metrics in merged.items():
            self._logger.log_metrics(metrics, step=step)

        self._log_buffer.clear()

    def on_start(self, step: int, metrics: dict[str, float], *args, **kwargs):
        """Called at training start: log rollout metrics, clear epoch buffer."""
        self._add_to_buffer(metrics, step)
        self._epoch_buffer.clear()

    def on_minibatch(self, metrics: dict[str, float], **kwargs):
        """Called after each minibatch."""
        if self.cfg.granularity == "minibatch":
            self._step += 1
            self._add_to_buffer(metrics, self._step)
        else:
            self._epoch_buffer.append(metrics)

    def on_epoch(self, metrics: dict[str, float] | None = None, *args, **kwargs):
        """Called after each epoch: log epoch-averaged metrics if configured."""
        if metrics:
            if self.cfg.granularity == "minibatch":
                # Diagnostics are transferred once per epoch and share the
                # final minibatch's step without incurring per-minibatch copies.
                self._add_to_buffer(metrics, self._step)
            else:
                self._epoch_buffer.append(metrics)
        if self.cfg.granularity == "epoch" and self._epoch_buffer:
            keys = set().union(*(entry.keys() for entry in self._epoch_buffer))
            aggregated = {
                key: float(np.mean([entry[key] for entry in self._epoch_buffer if key in entry]))
                for key in keys
            }
            self._step += 1
            self._add_to_buffer(aggregated, self._step)
            self._epoch_buffer.clear()

    def on_end(self, step: int, metrics: dict[str, float] | None, *args, **kwargs):
        """Called at training end: log final metrics, batch-averaged if configured."""
        if metrics:
            self._add_to_buffer(metrics, step)

        if self.cfg.granularity == "batch" and self._epoch_buffer:
            keys = set().union(*(entry.keys() for entry in self._epoch_buffer))
            aggregated = {
                key: float(np.mean([entry[key] for entry in self._epoch_buffer if key in entry]))
                for key in keys
            }
            self._step += 1
            self._add_to_buffer(aggregated, step)
            self._epoch_buffer.clear()

        self.flush()
