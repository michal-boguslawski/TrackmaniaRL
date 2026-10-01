"""Local filesystem backend for run metadata and model artifacts."""

import json
from pathlib import Path
from typing import Any

import torch as T
import yaml
from torch import nn

from rl_lib.tracking.artifacts import normalize_artifact_path
from rl_lib.tracking.base import MetricsLogger


class LocalArtifactLogger(MetricsLogger):
    """Persist model artifacts to a run-specific local directory.

    Artifact paths supplied by callbacks are preserved beneath ``root`` and
    receive a ``.pt`` suffix, except that any ``step_<digits>`` segment is
    zero-padded so artifact names sort in step order. Configurations,
    parameters, and final evaluation details are stored as small text files;
    scalar metrics remain a no-op.

    Attributes:
        root: Root directory for this run's artifacts.
    """

    def __init__(self, root: str | Path):
        """Initialize the logger.

        Args:
            root: Directory under which logger artifact paths are stored.
        """
        self.root = Path(root)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Ignore scalar metrics; this backend only stores model artifacts."""

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Save run parameters as JSON under the run's config directory.

        Args:
            parameters: Parameter names and JSON-serializable values.
        """
        self._metadata_file("config/parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        """Save the full run configuration as YAML.

        Args:
            config: Full JSON-compatible run configuration.
            artifact_file: Relative config file path within this run's artifacts.
        """
        self._metadata_file(artifact_file).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def log_evaluation(
        self,
        episodes: list[dict[str, float | int]],
        summary: dict[str, float],
        step: int,
        scope: str,
        config_artifact: str = "config/run_config.yaml",
    ) -> None:
        """Save per-episode and aggregate evaluation results as JSON.

        Args:
            episodes: Per-episode returns and lengths.
            summary: Aggregate evaluation metrics.
            step: Training step at which the evaluation ran.
            scope: Evaluation scope, such as ``final`` or ``periodic``.
            config_artifact: Run-relative path to the config used for evaluation.
        """
        evaluation = {
            "scope": scope,
            "step": step,
            "episodes": episodes,
            "summary": summary,
            "config_artifact": config_artifact,
        }
        self._metadata_file(f"evaluation/{scope}_results.json").write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Serialize a complete PyTorch model under the artifact path.

        Args:
            model: Model module to serialize.
            artifact_path: Relative path within this run's artifact directory.
            registered_model_name: Ignored; local artifacts are not registered.
        """
        T.save(model, self._artifact_file(artifact_path))

    def log_state_dict(
        self,
        state_dict: dict[str, Any],
        artifact_path: str = "checkpoints",
    ) -> None:
        """Serialize a model state dictionary under the artifact path.

        Args:
            state_dict: Model state dictionary to serialize.
            artifact_path: Relative path within this run's artifact directory.
        """
        T.save(state_dict, self._artifact_file(artifact_path))

    def _artifact_file(self, artifact_path: str) -> Path:
        """Create and return the local file path for an artifact."""
        relative_path = Path(normalize_artifact_path(artifact_path)).with_suffix(".pt")
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("artifact_path must be relative and stay within the logger root")

        artifact_file = self.root / relative_path
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        return artifact_file

    def _metadata_file(self, artifact_path: str) -> Path:
        """Create and return a safe run-relative metadata file path."""
        relative_path = Path(artifact_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("artifact_path must be relative and stay within the logger root")

        metadata_file = self.root / relative_path
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        return metadata_file
