"""Unified callback interface for training and rollout collection.

Defines hooks for PPO optimization and environment rollout collection.
CallbackList dispatches each event to its registered callbacks.
"""

from typing import Generic, TypeVar


class Callback:
    """Base callback for PPO training and rollout collection events.

    PPOTrainer invokes ``on_start``, ``on_minibatch``, ``on_epoch`` and
    ``on_end``. RolloutCollector invokes ``on_rollout_start``, ``on_env_step``,
    ``on_rollout_end`` and ``flush``. Unused hooks default to no-ops.
    """

    def on_start(self, *args, **kwargs) -> None:
        """Handle the trainer-start event."""

    def on_minibatch(self, *args, **kwargs) -> None:
        """Handle the trainer-minibatch event."""

    def on_end(self, *args, **kwargs) -> None:
        """Handle the trainer-end event."""

    def on_epoch(self, *args, **kwargs) -> None:
        """Handle the trainer-epoch event."""

    def on_rollout_start(self, *args, **kwargs) -> None:
        """Handle the collector-start event."""

    def on_env_step(self, *args, **kwargs) -> None:
        """Handle the collector-environment-step event."""

    def on_rollout_end(self, *args, **kwargs) -> None:
        """Handle the collector-end event."""

    def flush(self, *args, **kwargs) -> None:
        """Handle the collector-flush event."""


C = TypeVar("C", bound=Callback)


class CallbackList(Generic[C]):
    """Dispatch callback events in registration order.

    Args:
        callbacks: Callback instances to receive dispatched events.
    """

    def __init__(self, callbacks: list[C] | None = None):
        self._callbacks: list[C] = callbacks or []

    def __getattr__(self, name: str):
        """Return a dispatcher for a callback event."""

        def _dispatch(*args, **kwargs):
            for callback in self._callbacks:
                getattr(callback, name)(*args, **kwargs)

        return _dispatch
