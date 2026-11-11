from typing import Literal

from rl_lib.run_config import RolloutSettings
from rl_lib.training.ppo.config.network import PPONetworkConfig
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.settings.algorithm import AlgorithmSettings


class PPOAlgorithmSettings(AlgorithmSettings):
    name: Literal["ppo"]
    network: PPONetworkConfig
    trainer: PPOTrainerSettings
    buffer: RolloutSettings
