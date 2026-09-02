from logging import getLogger
from pathlib import Path

from rl_lib.training.callbacks.base import TrainingCallback
from rl_lib.agent import Agent


logger = getLogger(__name__)


class CheckpointsSaveCallback(TrainingCallback):
    def __init__(self, path: str, agent: Agent, intervals: int = 20):
        self._path = Path(path)
        self._agent = agent
        self._intervals = intervals
        self._cnt = 0
        self._path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CheckpointsSaveCallback created with intervals={self._intervals}")
        

    def on_end(self, *args, **kwargs) -> None:
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
