"""Base callback interfaces for training and rollout collection.

Defines abstract base classes for callbacks at two levels:
- TrainingCallback: called during PPO optimization (minibatch/epoch/start/end)
- CollectorCallback: called during environment rollout collection (step/start/end)

CallbackList provides a generic dispatcher that forwards calls to all registered
callbacks of the same type.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


class TrainingCallback(ABC):
    """Callbacks for PPO training loop events.

    Called by PPOTrainer at:
    - on_start: before epochs begin (receives step, metrics from full batch)
    - on_minibatch: after each minibatch step (receives metrics, step)
    - on_epoch: after each epoch completes
    - on_end: after all epochs (receives final metrics, step)
    """

    @abstractmethod
    def on_start(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_minibatch(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_end(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_epoch(self, *args, **kwargs) -> None: ...


class CollectorCallback(ABC):
    """Callbacks for rollout collection events.

    Called by RolloutCollector at:
    - on_rollout_start: before collection begins (receives config)
    - on_env_step: after each environment step (receives step index, info)
    - on_rollout_end: after collection completes
    - flush: periodic flush (e.g., when buffer fills and trainer runs)
    """

    @abstractmethod
    def on_rollout_start(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_env_step(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_rollout_end(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def flush(self, *args, **kwargs) -> None: ...


C = TypeVar("C", TrainingCallback, CollectorCallback)


class CallbackList(Generic[C]):
    """Dispatches method calls to a list of same-type callbacks.

    Uses __getattr__ to dynamically forward any method call to all
    registered callbacks. Unknown methods raise AttributeError.

    Args:
        callbacks: List of callback instances (TrainingCallback or CollectorCallback).
    """

    def __init__(self, callbacks: list[C] | None = None):
        self._callbacks: list[C] = callbacks or []

    def __getattr__(self, name: str):
        def _dispatch(*args, **kwargs):
            for cb in self._callbacks:
                getattr(cb, name)(*args, **kwargs)
        return _dispatch
