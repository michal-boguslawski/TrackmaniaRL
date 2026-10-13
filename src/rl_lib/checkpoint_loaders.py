"""Load local and MLflow model checkpoints into configured networks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import mlflow
import torch as T
from torch import nn

from rl_lib.tracking.artifacts import step_artifact_path


FINAL_MODEL_ARTIFACT = "final_model"


def _load_checkpoint_object(network: nn.Module, checkpoint: object) -> nn.Module:
    """Copy checkpoint model weights into a configured network."""
    while hasattr(checkpoint, "_orig_mod"):
        checkpoint = checkpoint._orig_mod
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint is neither a model nor a state dict")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model state dict")
    if state_dict and all(
        isinstance(key, str) and key.startswith("_orig_mod.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    nn.Module.load_state_dict(network, state_dict, strict=True)
    return network


class LocalCheckpointLoader:
    """Load a local PyTorch model or state-dict checkpoint.

    Full model files are deserialized with ``weights_only=False`` for backward
    compatibility, so only load files from trusted sources.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        """Configure the loader for one local checkpoint.

        Args:
            checkpoint_path: Local checkpoint file path.
        """
        self.checkpoint_path = Path(checkpoint_path)

    def load(self, network: nn.Module, device: T.device | str) -> nn.Module:
        """Load checkpoint weights into ``network`` and return it.

        Args:
            network: Configured model to receive the checkpoint weights.
            device: Device on which checkpoint tensors should be loaded.

        Returns:
            The same network with checkpoint weights loaded.

        Raises:
            ValueError: If the file does not contain a model or state dict.
            RuntimeError: If state-dict keys or tensor shapes are incompatible.
        """
        checkpoint = T.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        try:
            return _load_checkpoint_object(network, checkpoint)
        except ValueError as exc:
            raise ValueError(f"Invalid checkpoint {self.checkpoint_path}: {exc}") from exc


class MLflowCheckpointLoader:
    """Load a final model or selected state-dict checkpoint from an MLflow run."""

    def __init__(self, run_id: str, checkpoint_step: int | None = None) -> None:
        """Configure the loader for an MLflow run artifact.

        Args:
            run_id: MLflow run ID or ``runs:/<run_id>/...`` URI.
            checkpoint_step: Optional rollout step for a periodic state-dict
                checkpoint. When omitted, loads the ``final_model`` artifact.

        Raises:
            ValueError: If the run identifier is empty or step is negative.
        """
        value = run_id.strip()
        if value.startswith("runs:/"):
            value = value.removeprefix("runs:/").split("/", 1)[0]
        if not value:
            raise ValueError("MLflow run identifier cannot be empty")
        if checkpoint_step is not None and checkpoint_step < 0:
            raise ValueError("checkpoint_step must be non-negative")
        self.run_id = value
        self.checkpoint_step = checkpoint_step

    def load(self, network: nn.Module, device: T.device | str) -> nn.Module:
        """Load MLflow model weights into ``network`` and return it.

        Args:
            network: Configured model to receive the checkpoint weights.
            device: Device on which checkpoint tensors should be loaded.

        Returns:
            The same network with MLflow checkpoint weights loaded.

        Raises:
            ValueError: If the artifact does not contain a model or state dict.
            RuntimeError: If state-dict keys or tensor shapes are incompatible.
        """
        if self.checkpoint_step is not None:
            artifact_path = (
                f"{step_artifact_path('checkpoints', self.checkpoint_step)}"
                "/state_dict.pth"
            )
            checkpoint_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path=artifact_path,
            )
            return LocalCheckpointLoader(checkpoint_path).load(network, device)

        model = mlflow.pytorch.load_model(
            f"runs:/{self.run_id}/{FINAL_MODEL_ARTIFACT}",
            map_location=device,
        )
        return _load_checkpoint_object(network, model)
