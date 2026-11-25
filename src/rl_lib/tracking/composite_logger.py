from collections.abc import Callable
from logging import getLogger

from rl_lib.tracking.sinks.base import MetricsSink

logger = getLogger(__name__)


def _make_forwarder(name: str) -> Callable:
    def forward(self, *args, **kwargs):
        for sink in self._sinks:
            getattr(sink, name)(*args, **kwargs)  # exceptions propagate

    forward.__name__ = name
    return forward


class _SinkBroadcastMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__mro__:
            for name in getattr(base, "__abstractmethods__", ()):
                still_abstract = getattr(
                    getattr(cls, name, None), "__isabstractmethod__", False
                )
                if still_abstract:
                    setattr(cls, name, _make_forwarder(name))


class CompositeMetricsLogger(_SinkBroadcastMixin, MetricsSink):
    def __init__(self, sinks: list[MetricsSink] | None = None):
        self._sinks = sinks or []

    def add_sink(self, sink: MetricsSink) -> None:
        self._sinks.append(sink)
        logger.debug(f"Added metrics sink: {sink.__class__.__name__}")
