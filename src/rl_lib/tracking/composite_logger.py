"""Fan out tracking calls to a collection of metrics sinks."""

from collections.abc import Callable
from contextlib import ExitStack
from logging import getLogger
from types import TracebackType

from rl_lib.tracking.sinks.base import MetricsSink
from rl_lib.tracking.utils import run_status_for_exception

logger = getLogger(__name__)


def _make_forwarder(name: str) -> Callable[..., None]:
    """Create a method that forwards calls to each configured sink."""

    def forward(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Cannot log through a closed metrics logger.")
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
    """Broadcast tracking calls and coordinate sink shutdown.

    Sinks are closed in reverse registration order. Shutdown attempts every
    sink even if one fails, then raises the first failure.
    """

    def __init__(self, sinks: list[MetricsSink] | None = None):
        """Initialize the logger with optional sinks.

        Args:
            sinks: Tracking sinks to receive broadcast calls.
        """
        self._sinks = sinks or []
        self._closed = False

    def add_sink(self, sink: MetricsSink) -> None:
        """Register a sink before shutdown.

        Args:
            sink: Tracking sink to add.

        Raises:
            RuntimeError: If this logger has already been closed.
        """
        if self._closed:
            raise RuntimeError("Cannot add a sink to a closed metrics logger.")
        self._sinks.append(sink)
        logger.debug(f"Added metrics sink: {sink.__class__.__name__}")

    def close(self, status: str = "FINISHED") -> None:
        """Stop forwarding calls and close all registered sinks.

        The method is idempotent. It attempts to close every sink, even if an
        earlier sink raises. Any shutdown errors propagate with exception
        chaining handled by ``ExitStack``.

        Args:
            status: Final run status passed to sinks.
        """
        if self._closed:
            return
        self._closed = True
        with ExitStack() as stack:
            for sink in self._sinks:
                stack.callback(sink.close, status=status)

    def __enter__(self) -> "CompositeMetricsLogger":
        """Return this open logger for use as a context manager."""
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed metrics logger.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close sinks with a status matching the exception, if any.

        Returns:
            ``False`` so exceptions from the managed block propagate.
        """
        self.close(status=run_status_for_exception(exc_type, exc_val))
        return False
