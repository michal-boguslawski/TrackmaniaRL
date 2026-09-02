import socket
import uuid
from datetime import datetime, timezone

import mlflow
import mlflow.pytorch
import torch

from src.rl_lib.tracking.base import MetricsLogger


class MLflowLogger(MetricsLogger):
    def __init__(
        self,
        experiment_name: str = "MLflow Quickstart",
        run_name: str | None = None,
        registered_model_name: str | None = None,
    ):
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = self._deduce_run_name()

        self._registered_model_name = registered_model_name
        self._run = mlflow.start_run(run_name=run_name)

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

    def log_metrics(self, metrics: dict[str, float], step: int):
        mlflow.log_metrics(metrics, step=step)

    def log_parameters(self, parameters: dict[str, float]):
        mlflow.log_params({k: str(v) for k, v in parameters.items()})

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ):
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name or self._registered_model_name,
        )

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints"):
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

    def close(self):
        mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
