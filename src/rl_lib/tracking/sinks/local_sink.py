"""Console metrics logger for development/debugging.

Logs metrics and parameters to Python logging at DEBUG level.
Useful for local development without MLflow server.
"""

import json
from logging import getLogger
from pathlib import Path
from typing import Any

import torch as T
import yaml
from torch import nn

from rl_lib.paths import RunPaths
from rl_lib.tracking.sinks.base import MetricsSink

logger = getLogger(__name__)


class LocalMetricsSink(MetricsSink):
    """Simple local logger for metrics and parameters.

    Formats metrics with 6 decimal places and logs at DEBUG level.
    Parameters are logged as-is.
    """

    def __init__(self, run_paths: RunPaths):
        self.run_paths = run_paths

    @staticmethod
    def _save_json(dict_: dict, filepath: Path) -> None:
        filepath.write_text(
            json.dumps(dict_, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def _resolve_path(self, file_name: str) -> Path:
        file_path = self.run_paths.artifacts_dir / file_name
        file_path.parent.mkdir(exist_ok=True, parents=True)
        return file_path

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to local at DEBUG level."""
        formatted = {k: f"{v:.6f}" for k, v in metrics.items()}
        logger.debug(f"Step {step or ''}: {formatted}")

    def log_parameters(
        self,
        parameters: dict[str, Any],
    ) -> None:
        """Log parameters to local at DEBUG level."""
        formatted = {k: f"{v}" for k, v in parameters.items()}
        logger.debug(formatted)

        # filepath = self._resolve_path(artifact_path)
        # self._save_json(parameters, filepath)

    def log_config(
        self,
        config: dict[str, Any],
        artifact_path: str = "config/run_config.yaml",
    ) -> None:
        filepath = self._resolve_path(artifact_path)
        filepath.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

    def log_evaluation(
        self,
        results: dict[str, float | dict[str, float] | list[dict[str, float]]],
        artifact_path: str = "evaluation/final_results.json",
    ) -> None:
        filepath = self._resolve_path(artifact_path)
        self._save_json(results, filepath)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        extension: str = ".pt",
    ) -> None:
        filepath = self._resolve_path(artifact_path + extension)
        T.save(model, filepath)

    def log_state_dict(
        self,
        state_dict: dict[str, Any],
        artifact_path: str = "checkpoints/state_dict",
        extension: str = ".pt",
    ):
        filepath = self._resolve_path(artifact_path + extension)
        T.save(state_dict, filepath)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        pass
