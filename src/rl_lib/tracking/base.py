from abc import ABC, abstractmethod


class MetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
