"""Unified callback interface for training and rollout collection.

Defines hooks for PPO optimization and environment rollout collection.
CallbackList dispatches each event to its registered callbacks.
"""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, TypeVar


class CallbackEvent(StrEnum):
    TRAIN_START = "train_start"
    TRAIN_END = "train_end"
    EPOCH_END = "epoch_end"
    MINIBATCH_END = "minibatch_end"
    ROLLOUT_START = "rollout_start"
    ROLLOUT_END = "rollout_end"
    ENV_STEP = "env_step"


class Callback(ABC):
    @abstractmethod
    def on_event(
        self,
        event: CallbackEvent,
        **kwargs: Any,
    ) -> None:
        """Handle a training or rollout event."""

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered callback data."""


class CallbackList(Callback):
    """Dispatch callback events in registration order.

    Args:
        callbacks: Callback instances to receive dispatched events.
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        self._callbacks: list[Callback] = callbacks or []

    def on_event(
        self,
        event: CallbackEvent,
        **kwargs: Any,
    ) -> None:
        for callback in self._callbacks:
            callback.on_event(event, **kwargs)

    def flush(self) -> None:
        for callback in self._callbacks:
            callback.flush()
