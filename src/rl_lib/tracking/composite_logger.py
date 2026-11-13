from logging import getLogger

from rl_lib.tracking.sinks.base import MetricsSink

logger = getLogger(__name__)


class CompositeMetricsLogger(MetricsSink):
    def __init__(self, sinks: list[MetricsSink] | None = None):
        self._sinks = sinks or []
        # self._log_buffer: list[tuple[dict[str, float], int]] = []

    def add_sink(self, sink: MetricsSink) -> None:
        self._sinks.append(sink)
        logger.debug(f"Added metrics sink: {sink.__class__.__name__}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to all sinks, buffering for later flush."""
        for sink in self._sinks:
            sink.log_metrics(metrics, step=step)
        # self._log_buffer.append((metrics, step))

    def log_parameters(self, parameters: dict[str, int | float | str]) -> None:
        for sink in self._sinks:
            sink.log_parameters(parameters)

    # def flush(self) -> None:
    #     """Flush buffered metrics to all sinks, merging by step."""
    #     merged: dict[int, dict[str, float]] = {}
    #     for metrics, step in self._log_buffer:
    #         merged.setdefault(step, {}).update(metrics)

    #     for step, metrics in merged.items():
    #         for sink in self._sinks:
    #             sink.log_metrics(metrics, step=step)

    #     self._log_buffer.clear()
