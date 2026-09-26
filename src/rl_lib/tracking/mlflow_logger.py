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
    is a deliberate clean exit, so it still counts as `FINISHED`."""
    if exc_type is None:
        return "FINISHED"
    if issubclass(exc_type, KeyboardInterrupt):
        return "KILLED"
    if issubclass(exc_type, SystemExit) and exc_val is not None and getattr(exc_val, "code", None) in (0, None):
        return "FINISHED"
    return "FAILED"


class MLflowLogger(MetricsLogger):
    def __init__(
        self,
        experiment_name: str = "MLflow Quickstart",
        run_name: str | None = None,
        registered_model_name: str | None = None,
        log_system_metrics: bool = True,
    ):
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = self._deduce_run_name()

        self._registered_model_name = registered_model_name
        self._run = mlflow.start_run(run_name=run_name, log_system_metrics=log_system_metrics)

    @staticmethod
    def _deduce_run_name() -> str:
        """
        <hostname>-<UTC timestamp>-<short uuid>
        e.g. "gpu-node-01-20260901T142230Z-4f9a1c"
        Timestamp keeps runs sortable in the UI; short uuid avoids
        collisions when multiple runs start in the same second
        (e.g. parallel seeds / grid search).
        """
        host = socket.gethostname().split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:6]
        return f"{host}-{ts}-{short_id}"

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_parameters(self, parameters: dict[str, float]) -> None:
        mlflow.log_params({k: str(v) for k, v in parameters.items()})

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        mlflow.log_dict(config, artifact_file)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name or self._registered_model_name,
        )

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def close(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Returns None so the exception keeps propagating: the process exit
        # status and the run status must agree, and the caller relies on this
        # `with` block not swallowing a failure.
        self.close(status=run_status_for_exception(exc_type, exc_val))
