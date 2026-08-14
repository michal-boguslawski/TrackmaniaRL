from abc import ABC, abstractmethod
from typing import Generic, TypeVar


class TrainingCallback(ABC):
    @abstractmethod
    def on_start(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_minibatch(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_end(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_epoch(self, *args, **kwargs) -> None: ...


class CollectorCallback(ABC):
    @abstractmethod
    def on_rollout_start(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_env_step(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_rollout_end(self, *args, **kwargs) -> None: ...


C = TypeVar("C", TrainingCallback, CollectorCallback)


class CallbackList(Generic[C]):
    def __init__(self, callbacks: list[C] | None = None):
        self._callbacks: list[C] = callbacks or []

    def __getattr__(self, name: str):
        def _dispatch(*args, **kwargs):
            for cb in self._callbacks:
                getattr(cb, name)(*args, **kwargs)
        return _dispatch
