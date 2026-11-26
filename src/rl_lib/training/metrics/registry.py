from collections.abc import Callable
from typing import TypeVar

ContextT = TypeVar("ContextT")
type MetricFunction[ContextT] = Callable[[ContextT], dict[str, float]]


class MetricRegistry[ContextT]:
    def __init__(self, prefix: str | None = None) -> None:
        self._functions: list[MetricFunction[ContextT]] = []
        self.prefix = prefix

    def register(self, function: MetricFunction[ContextT]) -> MetricFunction[ContextT]:
        self._functions.append(function)
        return function

    def calculate(self, context: ContextT) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for function in self._functions:
            metrics |= function(context)
        return {
            f"{self.prefix}/{name}" if self.prefix else name: value
            for name, value in metrics.items()
        }
