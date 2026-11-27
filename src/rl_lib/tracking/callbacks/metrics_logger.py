"""Log training metrics to a configured sink at the selected granularity.

Minibatch metrics can be logged individually, averaged at each epoch boundary,
or averaged across the full training batch and logged at the training boundary.
"""

from __future__ import annotations

from typing import Any

import torch as T

from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.config.callbacks.metrics_logger import MetricsCallbackSettings
from rl_lib.tracking.metric_values import metric_values_to_scalars
from rl_lib.tracking.sinks.base import MetricsSink


class MetricsLoggingCallback(Callback):
    """Log scalar training metrics at minibatch, epoch, or training granularity.

    Attributes:
        _logger (MetricsSink): Backend such as the console or MLflow sink.
        cfg (MetricsCallbackSettings): Configured logging granularity.
        _average_buffer (dict[str, list[T.Tensor]]): Metrics waiting to be averaged.
        _log_buffer (list[tuple[dict[str, float], int]]): Independent records waiting to be logged.
        _step (int): Next fallback step when an event provides no step.
    """

    def __init__(
        self,
        logger: MetricsSink,
        config: MetricsCallbackSettings,
        prefix: str = "train",
        *args,
        **kwargs,
    ):
        """Initialize the callback.

        Args:
            logger: Sink that writes metrics to a backend.
            config: Settings selecting minibatch, epoch, or training-level logging.
        """
        self._logger = logger
        self.cfg = config
        self._average_buffer: dict[str, list[T.Tensor]] = {}
        self._log_buffer: list[tuple[dict[str, float], int]] = []
        self.prefix = prefix

    def flush(self) -> None:
        """Write every pending metrics record without merging equal steps."""
        for metrics, step in self._log_buffer:
            self._logger.log_metrics(
                {f"{self.prefix}/{metric}": value for metric, value in metrics.items()},
                step,
            )
        self._log_buffer.clear()

    def on_event(
        self,
        event: CallbackEvent,
        metrics: dict[str, T.Tensor] | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Collect metrics and emit them at the configured event boundary.

        Args:
            event: Callback lifecycle event being dispatched.
            metrics: Scalar metrics associated with the event, if any.
            step: Backend step for the event. A callback-local step is used if
                omitted.
            **kwargs: Additional event data, ignored by this callback.
        """
        if event in [
            CallbackEvent.MINIBATCH_END,
            CallbackEvent.EPOCH_END,
            CallbackEvent.TRAIN_END,
            CallbackEvent.TRAIN_START,
        ]:
            if metrics:
                for name, value in metrics.items():
                    if not isinstance(value, T.Tensor):
                        raise TypeError(
                            f"Training metric {name!r} must be a tensor, "
                            f"got {type(value).__name__}"
                        )
                    if value.requires_grad:
                        raise ValueError(
                            f"Training metric {name!r} must be detached before logging"
                        )
                    self._average_buffer.setdefault(name, []).append(value)

            self._collect(step, event)

    def _collect(
        self,
        step: int | None,
        boundary: CallbackEvent,
    ) -> None:
        """Accumulate event metrics and queue an average at the configured boundary."""

        if self.cfg.granularity == boundary:
            self._queue(self._average(self._average_buffer), step)
            self._average_buffer.clear()

    @staticmethod
    def _average(buffer: dict[str, list[T.Tensor]]) -> dict[str, float]:
        """Average metrics on-device, then transfer the results together."""
        averaged: dict[str, T.Tensor] = {}
        for name, values in buffer.items():
            for value in values:
                if value.ndim != 0:
                    raise ValueError(
                        f"Metric {name!r} must be a zero-dimensional tensor, "
                        f"got shape {tuple(value.shape)}"
                    )
                if value.device != values[0].device or value.dtype != values[0].dtype:
                    raise ValueError(
                        f"Values for metric {name!r} must share a device and "
                        "dtype across logging events"
                    )
            averaged[name] = T.stack(values).mean()

        return metric_values_to_scalars(averaged)

    def _queue(self, metrics: dict[str, float] | None, step: int | None) -> None:
        """Queue a non-empty metric record and resolve its backend step."""
        if not metrics or (step is None):
            return

        self._log_buffer.append((dict(metrics), step))
