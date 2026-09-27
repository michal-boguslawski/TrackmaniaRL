"""Collector callback: saves final model to disk and logs to tracking backends.

Runs at the end of rollout collection (after all training steps complete).
Saves the full network (architecture + weights) and state dict separately.
"""

from logging import getLogger
from pathlib import Path
from typing import Iterable

import torch as T

from rl_lib.agent import Agent
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback


logger = getLogger(__name__)


class FinalModelSaveCallback(CollectorCallback):
    """Save the complete model locally and forward it to attached loggers.

    On rollout end:
    - Creates checkpoint directory if needed
    - Saves full nn.Module to final_model.pt
    - Logs model artifact and state_dict to all metrics loggers
    """

    def __init__(
        self,
        agent: Agent,
        folder: str | Path,
        metrics_loggers: Iterable[MetricsLogger] = (),
    ):
        """Initialize the callback.

        Args:
            agent: Training agent whose network will be saved.
            folder: Directory to save final_model.pt.
            metrics_loggers: Tracking backends for artifact logging.
        """
        self._agent = agent
        self._path = Path(folder) / "final_model.pt"
        self._metrics_loggers = list(metrics_loggers)

    def on_rollout_end(self, *args, **kwargs) -> None:
        """Save model and log artifacts when rollout completes."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        T.save(self._agent.network, self._path)
        logger.info("Saved final model to %s", self._path)

        for metrics_logger in self._metrics_loggers:
            metrics_logger.log_model(
                self._agent.network,
                artifact_path="final_model",
            )
            metrics_logger.log_state_dict(
                self._agent.network.state_dict(),
                artifact_path="final_state_dict",
            )

    def on_rollout_start(self, *args, **kwargs) -> None:
        pass

    def on_env_step(self, *args, **kwargs) -> None:
        pass

    def flush(self, *args, **kwargs) -> None:
        pass
