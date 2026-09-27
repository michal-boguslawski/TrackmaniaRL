"""MLflow tracking backend with context manager for run lifecycle.

Provides MetricsLogger implementation using MLflow Python API. Includes
context manager support for automatic run finalization with proper status
mapping (FINISHED/KILLED/FAILED).
"""

import socket
import uuid
from datetime import datetime, timezone

import mlflow
import mlflow.pytorch
import torch
from typing import Self

from rl_lib.tracking.base import MetricsLogger


def run_status_for_exception(exc_type: type[BaseException] | None, exc_val: BaseException | None = None) -> str:
    """Map the exception leaving a run's `with` block onto an MLflow run status.

    A run that ends because it was interrupted is `KILLED` rather than
    `FAILED`, so the UI distinguishes a stop from a crash. `SystemExit(0)`
    is a deliberate clean exit, so it still counts as `FINISHED`.

    Args:
        exc_type: Exception type from __exit__ (None if no exception).
        exc_val: Exception value.

    Returns:
        MLflow run status string: FINISHED, KILLED, or FAILED.
    """
    if exc_type is None:
        return "FINISHED"
    if issubclass(exc_type, KeyboardInterrupt):
        return "KILLED"
    if issubclass(exc_type, SystemExit) and exc_val is not None and getattr(exc_val, "code", None) in (0, None):
        return "FINISHED"
    return "FAILED"


class MLflowLogger(MetricsLogger):
    """MLflow tracking backend for metrics, params, models, and artifacts.

    Creates an MLflow run on initialization. Supports context manager
    protocol for automatic cleanup with proper status mapping.

    Attributes:
        _registered_model_name: Optional model registry name.
        _run: Active MLflow run object.
    """

    def __init__(
        self,
        experiment_name: str = "MLflow Quickstart",
        run_name: str | None = None,
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
        self._run = mlflow.start_run(run_name=run_name, log_system_metrics=log_system_metrics)

    @staticmethod
    def _deduce_run_name() -> str:
        """Generate unique run name: <hostname>-<UTC timestamp>-<short uuid>.

        Format keeps runs sortable in UI; short uuid avoids collisions
        when multiple runs start in same second (parallel seeds/grid search).

        Returns:
            Run name string.
        """
        host = socket.gethostname().split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:6]
        return f"{host}-{ts}-{short_id}"

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_parameters(self, parameters: dict[str, object]) -> None:
        """Log parameters to MLflow (values converted to strings)."""
        mlflow.log_params({k: str(v) for k, v in parameters.items()})

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        """Log full config dict as YAML artifact."""
        mlflow.log_dict(config, artifact_file)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log PyTorch model to MLflow with optional registry."""
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name or self._registered_model_name,
        )

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        """Log state dict as MLflow artifact."""
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log local file (e.g., video) as MLflow artifact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def close(self, status: str = "FINISHED") -> None:
        """End MLflow run with given status."""
        mlflow.end_run(status=status)

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close run with status derived from exception (if any).

        Does not suppress exceptions; returns None so exception propagates.
        Process exit status and MLflow run status must agree.
        """
        self.close(status=run_status_for_exception(exc_type, exc_val))
