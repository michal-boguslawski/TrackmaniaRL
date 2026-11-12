from __future__ import annotations

from typing import Literal

from rl_lib.networks.settings.actor import ActorSettings
from rl_lib.networks.settings.critic import VCriticSettings
from rl_lib.networks.settings.encoder import EncoderSettings
from rl_lib.training.settings.network import NetworkSettings


class PPONetworkSettings(NetworkSettings):
    encoder: EncoderSettings
    actor: ActorSettings
    critic: VCriticSettings


class PPOAlgorithmSettings(NetworkSettings):
    """PPO algorithm hyperparameters."""

    name: Literal["ppo"]
