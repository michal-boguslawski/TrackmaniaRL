"""Training callback: saves model checkpoints at regular intervals.

Saves agent network state dict to disk every N training updates (where
an "update" = one call to PPOTrainer.train(), i.e., once per full buffer).
"""

from logging import getLogger
from pathlib import Path

from rl_lib.training.callbacks.base import TrainingCallback
from rl_lib.run_config import CheckpointCallbackSettings
from rl_lib.agent import Agent


logger = getLogger(__name__)


class CheckpointsSaveCallback(TrainingCallback):
    """Save model checkpoints at regular training intervals.

    Attributes:
        _path: Directory for checkpoint files.
        _agent: Training agent to save.
        cfg: CheckpointCallbackSettings with folder and interval.
        _intervals: Number of updates between checkpoints.
        _cnt: Counter of completed updates.
    """

    def __init__(self, agent: Agent, config: CheckpointCallbackSettings):
        """Initialize checkpoint callback.

        Args:
            agent: Training agent whose state will be saved.
            config: Checkpoint settings (folder path, interval in updates).
        """
        self._path = Path(config.folder)
        self._agent = agent
        self.cfg = config
        self._intervals = config.interval
        self._cnt = 0
        self._path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CheckpointsSaveCallback created with intervals={self._intervals}")

    def on_end(self, *args, **kwargs) -> None:
        """Called after each PPOTrainer.train() call (one buffer's worth of updates)."""
        self._cnt += 1
        if self._cnt % self._intervals == 0:
            logger.debug("Saving checkpoint")
            self._agent.save_state_dict(self._path / f"checkpoint_{self._cnt}.pt")

    def on_start(self, *args, **kwargs) -> None:
        pass

    def on_minibatch(self, *args, **kwargs) -> None:
        pass

    def on_epoch(self, *args, **kwargs) -> None:
        pass
