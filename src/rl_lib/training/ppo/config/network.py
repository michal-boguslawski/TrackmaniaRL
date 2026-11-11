from __future__ import annotations

from typing import Literal

from rl_lib.networks.config import ActorSettings, EncoderSettings, VCriticSettings
from rl_lib.training.settings.network import NetworkSettings


class PPONetworkSettings(NetworkSettings):
    encoder: EncoderSettings
    actor: ActorSettings
    critic: VCriticSettings


class PPOAlgorithmSettings(NetworkSettings):
    """PPO algorithm hyperparameters."""

    name: Literal["ppo"]
