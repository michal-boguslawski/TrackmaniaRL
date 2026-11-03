from dataclasses import dataclass
from typing import Literal, Protocol

import torch as T

from rl_lib.run_config import RunConfig
from rl_lib.training.callbacks.base import Callback
from rl_lib.training.transitions import Transition


@dataclass(frozen=True)
class BuildContext:
    config: RunConfig
    device: T.device
    verbosity: Literal[0, 1, 2]
    obs_shape: tuple
    action_shape: tuple


class Trainer(Protocol):
    def on_run_start(self) -> None: ...
    def observe(self, t: Transition) -> None: ...
    def maybe_update(self, global_step: int) -> None: ...
    # def make_eval_policy(self) -> Policy: ...  # shared weights, fresh History
    def set_callbacks(self, callbacks: list[Callback]) -> None: ...
    def config(self) -> dict: ...
