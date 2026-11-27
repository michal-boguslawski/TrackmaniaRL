from collections.abc import Callable
from typing import TypeVar

import torch as T

ContextT = TypeVar("ContextT")
type MetricFunction[ContextT] = Callable[[ContextT], dict[str, T.Tensor]]


class MetricRegistry[ContextT]:
    def __init__(self, prefix: str | None = None) -> None:
        self._functions: list[MetricFunction[ContextT]] = []
        self.prefix = prefix

    def register(self, function: MetricFunction[ContextT]) -> MetricFunction[ContextT]:
        self._functions.append(function)
        return function

    def calculate(self, context: ContextT) -> dict[str, T.Tensor]:
        metrics: dict[str, T.Tensor] = {}
        for function in self._functions:
            metrics |= function(context)
        return {
            f"{self.prefix}/{name}" if self.prefix else name: value
            for name, value in metrics.items()
        }
