from typing import Literal

from rl_lib.config_base import StrictModel
from rl_lib.training.ppo.config.network import PPONetworkConfig
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings


class PPOAlgorithmConfig(StrictModel):
    name: Literal["ppo"]
    network: PPONetworkConfig
    trainer: PPOTrainerSettings
