from dataclasses import dataclass
from typing import Literal, Protocol

import torch as T

from rl_lib.run_config import RolloutSettings, RunConfig, RunSettings


@dataclass(frozen=True)
class BuildContext:
    config: RunConfig
    device: T.device
    verbosity: Literal[0, 1, 2]
    obs_shape: tuple
    action_shape: tuple


class Trainer(Protocol):
    """Interface required by the rollout collector and training entrypoint."""

    @property
    def stack_size(self) -> int:
        """Temporal observation window size."""
        ...

    @property
    def device(self) -> T.device:
        """Device used for policy and value computations."""
        ...

    def setup_train(self, run: RunSettings, rollout: RolloutSettings) -> None:
        """Initialize training schedules from run and rollout settings."""
        ...

    def act(
        self, observation: T.Tensor, done: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """Choose actions and return their log probabilities and values."""
        ...

    def bootstrap_value(self, observation: T.Tensor) -> T.Tensor:
        """Estimate values for truncated transitions."""
        ...

    def train(
        self,
        batch: dict[str, T.Tensor],
        epochs: int,
        minibatch_size: int,
        training_step: int,
    ) -> None:
        """Update the policy/value model from a collected batch."""
        ...

    def config(self) -> dict[str, int | float | str]:
        """Return trainer settings suitable for experiment logging."""
        ...

    def network_config(self) -> dict[str, int | float | str]:
        """Return network settings suitable for experiment logging."""
        ...
