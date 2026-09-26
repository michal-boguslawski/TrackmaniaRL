from logging import getLogger
from pathlib import Path

import torch as T

from rl_lib.agent import Agent
from rl_lib.tracking.mlflow_logger import MLflowLogger
from rl_lib.training.callbacks.base import CollectorCallback


logger = getLogger(__name__)


class FinalModelSaveCallback(CollectorCallback):
    """Save the complete model locally and log the model and state dict to MLflow."""

    def __init__(
        self,
        agent: Agent,
        folder: str | Path,
        mlflow_logger: MLflowLogger | None = None,
    ):
        self._agent = agent
        self._path = Path(folder) / "final_model.pt"
        self._mlflow_logger = mlflow_logger

    def on_rollout_end(self, *args, **kwargs) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        T.save(self._agent.network, self._path)
        logger.info("Saved final model to %s", self._path)

        if self._mlflow_logger is not None:
            self._mlflow_logger.log_model(
                self._agent.network,
                artifact_path="final_model",
            )
            self._mlflow_logger.log_state_dict(
                self._agent.network.state_dict(),
                artifact_path="final_state_dict",
            )

    def on_rollout_start(self, *args, **kwargs) -> None:
        pass

    def on_env_step(self, *args, **kwargs) -> None:
        pass

    def flush(self, *args, **kwargs) -> None:
        pass
