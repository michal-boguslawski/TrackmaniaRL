"""Logger-backed periodic and final model checkpoint callback.

The callback logs state dictionaries at rollout-step intervals and records the
complete model plus final state dictionary when rollout collection finishes.
"""

from logging import getLogger

from torch import nn

from rl_lib.tracking.artifacts import step_artifact_path
from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.config.callbacks.model_checkpoint import (
    ModelCheckpointCallbackSettings,
)
from rl_lib.tracking.sinks.base import MetricsSink

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
        logger: MetricsSink,
        config: ModelCheckpointCallbackSettings,
        modules: nn.Module | None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the callback.

        Args:
            agent: Training agent whose network will be logged.
            metrics_loggers: Tracking backends receiving logged artifacts.
            interval: Rollout-step interval for state-dict checkpoints. If
                ``None``, only final artifacts are logged.
        """
        if modules is None:
            raise TypeError("Missing modules as nn.Module")
        self._modules = modules
        self._logger = logger
        self.cfg = config

    def on_event(self, event: CallbackEvent, step: int, *args, **kwargs) -> None:
        if event is CallbackEvent.ENV_STEP and step % self.cfg.interval == 0:
            file_name = f"checkpoints/step_{step}"
            self._logger.log_state_dict(self._modules.state_dict(), file_name)
            logger.debug(f"State dict saved at {file_name}")

        if event is CallbackEvent.ROLLOUT_END:
            self._logger.log_model(self._modules, "final_model")
            logger.debug("Final model saved at final_model")
            self._logger.log_state_dict(self._modules.state_dict(), "final_state_dict")
            logger.debug("Final state dict saved at final_state_dict")

    def flush(self) -> None:
        pass
