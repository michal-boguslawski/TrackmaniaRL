"""Abstract base class for metrics tracking backends.

Defines the interface for logging metrics, parameters, configs, models,
and artifacts. Concrete implementations (MLflowLogger, ConsoleMetricsLogger)
provide backend-specific storage.
"""

from abc import ABC, abstractmethod
from typing import Any


class MetricsLogger(ABC):
    """Base interface for experiment tracking backends.

    All methods are no-op by default; subclasses implement backend-specific
    logging (MLflow, TensorBoard, console, etc.).
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at a training step.

        Args:
            metrics: Dict of metric name -> value.
            step: Global step (e.g., environment steps or minibatch count).
        """
        ...

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Log hyperparameters/configuration (optional).

        Args:
            parameters: Dict of parameter name -> value.
        """
        pass

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        """Log full configuration as an artifact (optional).

        Args:
            config: Full configuration dict.
            artifact_file: Path within artifact store.
        """
        pass

    def log_evaluation(
        self,
        episodes: list[dict[str, float | int]],
        summary: dict[str, float],
        step: int,
        scope: str,
        config_artifact: str = "config/run_config.yaml",
    ) -> None:
        """Persist evaluation details when supported by the backend.

        Args:
            episodes: Per-episode returns and lengths.
            summary: Aggregate evaluation metrics.
            step: Training step at which the evaluation ran.
            scope: Evaluation scope, such as ``final`` or ``periodic``.
            config_artifact: Run-relative path to the config used for evaluation.
        """
        pass

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log model artifact (optional).

        Args:
            model: Model object (e.g., nn.Module).
            artifact_path: Path within artifact store.
            registered_model_name: Optional name for model registry.
        """
        pass

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        """Log model state dict (optional).

        Args:
            state_dict: Model state dict.
            artifact_path: Path within artifact store.
        """
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact (optional).

        Args:
            local_path: Path to local file.
            artifact_path: Destination path in artifact store.
        """
        pass
