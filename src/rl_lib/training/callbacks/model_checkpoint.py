"""Logger-backed periodic and final model checkpoint callback.

The callback logs state dictionaries at rollout-step intervals and records the
complete model plus final state dictionary when rollout collection finishes.
"""

from logging import getLogger
from typing import Iterable

from rl_lib.agent import Agent
from rl_lib.tracking.artifacts import step_artifact_path
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import Callback


logger = getLogger(__name__)


class ModelCheckpointCallback(Callback):
    """Log periodic state dictionaries and final model artifacts.

    Periodic checkpoints are emitted by the rollout collector after the
    configured number of vector-environment steps. Final artifacts are logged
    once collection ends.

    Attributes:
        _agent: Agent providing the network being tracked.
        _metrics_loggers: Backends receiving checkpoint and model artifacts.
        _interval: Number of rollout steps between periodic checkpoints, or
            ``None`` to log only final artifacts.
    """

    def __init__(
        self,
        agent: Agent,
        metrics_loggers: Iterable[MetricsLogger],
        interval: int | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            agent: Training agent whose network will be logged.
            metrics_loggers: Tracking backends receiving logged artifacts.
            interval: Rollout-step interval for state-dict checkpoints. If
                ``None``, only final artifacts are logged.
        """
        self._agent = agent
        self._metrics_loggers = list(metrics_loggers)
        if interval is not None and interval <= 0:
            raise ValueError("interval must be positive when configured")
        self._interval = interval
        self._next_checkpoint_step: int | None = interval

    def on_env_step(self, step: int, *args, **kwargs) -> None:
        """Log a checkpoint when the next rollout-step interval is reached."""
        if self._interval is None or self._next_checkpoint_step is None:
            return

        completed_steps = step + 1
        if completed_steps < self._next_checkpoint_step:
            return

        state_dict = self._agent.network.state_dict()
        artifact_path = step_artifact_path("checkpoints", completed_steps)
        for metrics_logger in self._metrics_loggers:
            metrics_logger.log_state_dict(state_dict, artifact_path=artifact_path)

        intervals_elapsed = (
            (completed_steps - self._next_checkpoint_step) // self._interval + 1
        )
        self._next_checkpoint_step += intervals_elapsed * self._interval
        logger.info("Logged model checkpoint at rollout step %s", completed_steps)

    def on_rollout_end(self, *args, **kwargs) -> None:
        """Log the final model and its state dictionary."""
        network = self._agent.network
        state_dict = network.state_dict()
        for metrics_logger in self._metrics_loggers:
            metrics_logger.log_model(network, artifact_path="final_model")
            metrics_logger.log_state_dict(
                state_dict,
                artifact_path="final_state_dict",
            )
