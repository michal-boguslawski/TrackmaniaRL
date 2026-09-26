from abc import ABC, abstractmethod
from typing import Any


class MetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Optional parameter logging; unsupported by default."""
        pass

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        """Optional config artifact logging; unsupported by default."""
        pass

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Optional model logging; unsupported by default."""
        pass

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        """Optional state-dict logging; unsupported by default."""
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Optional: log a local file (video, image, etc.) as a run artifact.
        No-op by default; override in loggers that support artifact storage."""
        pass
