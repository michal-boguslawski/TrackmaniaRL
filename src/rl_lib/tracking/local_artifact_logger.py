"""Local filesystem backend for model and state-dictionary artifacts."""

from pathlib import Path
from typing import Any

import torch as T
from torch import nn

from rl_lib.tracking.artifacts import normalize_artifact_path
from rl_lib.tracking.base import MetricsLogger


class LocalArtifactLogger(MetricsLogger):
    """Persist model artifacts to a run-specific local directory.

    Artifact paths supplied by callbacks are preserved beneath ``root`` and
    receive a ``.pt`` suffix, except that any ``step_<digits>`` segment is
    zero-padded so artifact names sort in step order. Scalar metrics and other
    optional logger methods are intentionally no-ops; use a metrics backend
    alongside this logger.

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
