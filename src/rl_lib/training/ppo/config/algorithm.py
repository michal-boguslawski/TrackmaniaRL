from typing import Literal

from rl_lib.buffers.settings.rollout import RolloutSettings
from rl_lib.training.ppo.config.network import PPONetworkConfig
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.settings.algorithm import AlgorithmConfig


class PPOAlgorithmConfig(AlgorithmConfig):
    name: Literal["ppo"]
    network: PPONetworkConfig
    trainer: PPOTrainerSettings
    buffer: RolloutSettings
