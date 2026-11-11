from typing import Literal

from rl_lib.run_config import RolloutSettings
from rl_lib.training.settings.base import StrictModel
from rl_lib.training.settings.network import NetworkSettings
from rl_lib.training.settings.trainer import TrainerSettings


class AlgorithmSettings(StrictModel):
    name: Literal["ppo"]
    network: NetworkSettings
    trainer: TrainerSettings
    buffer: RolloutSettings
