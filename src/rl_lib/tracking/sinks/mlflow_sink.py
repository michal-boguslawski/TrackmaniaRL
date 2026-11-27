"""MLflow tracking backend with context manager for run lifecycle.

Provides MetricsLogger implementation using MLflow Python API. Includes
context manager support for automatic run finalization with proper status
mapping (FINISHED/KILLED/FAILED).
"""

import socket
import uuid
from datetime import UTC, datetime
from typing import Self

import mlflow
import mlflow.pytorch
import torch

from rl_lib.tracking.artifacts import normalize_artifact_path
from rl_lib.tracking.sinks.base import MetricsSink
from rl_lib.tracking.utils import run_status_for_exception


class MLflowMetricsSink(MetricsSink):
    """MLflow tracking backend for metrics, params, models, and artifacts.

    Creates an MLflow run on initialization. Supports context manager
    protocol for automatic cleanup with proper status mapping. Artifact paths
    have any ``step_<digits>`` segment zero-padded, so step-named artifacts
    stay in step order.

    Attributes:
        _registered_model_name: Optional model registry name.
        _run: Active MLflow run object.
    """

    def __init__(
        self,
        experiment_name: str = "MLflow Quickstart",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        registered_model_name: str | None = None,
        log_system_metrics: bool = True,
    ):
        """Initialize MLflow logger and start run.

        Args:
            experiment_name: MLflow experiment name (created if not exists).
            run_name: Run name (auto-generated if None).
            registered_model_name: Optional model registry name.
            log_system_metrics: Enable MLflow system metrics logging.
        """
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = self._deduce_run_name()

        self._registered_model_name = registered_model_name
        self._run = mlflow.start_run(
            run_name=run_name, tags=tags, log_system_metrics=log_system_metrics
        )
        self._closed = False

    def _ensure_open(self) -> None:
        """Reject logging calls after this sink has ended its run."""
        if self._closed:
            raise RuntimeError("Cannot log through a closed MLflow sink.")

    @staticmethod
    def _deduce_run_name() -> str:
        """Generate unique run name: <hostname>-<UTC timestamp>-<short uuid>.

        Format keeps runs sortable in UI; short uuid avoids collisions
        when multiple runs start in same second (parallel seeds/grid search).

        Returns:
            Run name string.
        """
        host = socket.gethostname().split(".")[0]
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:6]
        return f"{host}-{ts}-{short_id}"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        self._ensure_open()
        mlflow.log_metrics(metrics, step=step)

    def log_parameters(self, parameters: dict[str, object]) -> None:
        """Log parameters to MLflow (values converted to strings)."""
        self._ensure_open()
        mlflow.log_params({k: str(v) for k, v in parameters.items()})

    def log_config(
        self, config: dict, artifact_file: str = "config/run_config.yaml"
    ) -> None:
        """Log full config dict as YAML artifact."""
        self._ensure_open()
        mlflow.log_dict(config, artifact_file)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model.pt",
        registered_model_name: str | None = None,
    ) -> None:
        """Log PyTorch model to MLflow with optional registry."""
        self._ensure_open()
        mlflow.pytorch.log_model(
            model,
            name=artifact_path,
            registered_model_name=registered_model_name or self._registered_model_name,
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
            pip_requirements=[f"torch=={torch.__version__.split('+')[0]}", "mlflow"],
        )

    def log_state_dict(
        self, state_dict: dict, artifact_path: str = "checkpoints/step"
    ) -> None:
        """Log state dict as MLflow artifact."""
        self._ensure_open()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log local file (e.g., video) as MLflow artifact."""
        self._ensure_open()
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_evaluation(
        self,
        results: dict[str, float | dict[str, float] | list[dict[str, float]]],
        artifact_path: str = "evaluation/final_results.json",
    ) -> None:
        """Log evaluation results as a JSON artifact."""
        self._ensure_open()
        mlflow.log_dict(results, artifact_path)

    def close(self, status: str = "FINISHED") -> None:
        """End the owned MLflow run once with the given status.

        Args:
            status: Final MLflow run status.
        """
        if self._closed:
            return
        mlflow.end_run(status=status)
        self._closed = True

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close run with status derived from exception (if any).

        Does not suppress exceptions; returns None so exception propagates.
        Process exit status and MLflow run status must agree.
        """
        self.close(status=run_status_for_exception(exc_type, exc_val))
