from abc import ABC, abstractmethod


class MetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Optional: log a local file (video, image, etc.) as a run artifact.
        No-op by default; override in loggers that support artifact storage."""
        pass
