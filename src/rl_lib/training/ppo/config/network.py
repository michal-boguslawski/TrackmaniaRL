from __future__ import annotations

from rl_lib.networks.settings.actor import ActorSettings
from rl_lib.networks.settings.critic import VCriticSettings
from rl_lib.networks.settings.encoder import EncoderConfig
from rl_lib.training.settings.network import NetworkConfig


class PPONetworkConfig(NetworkConfig):
    encoder: EncoderConfig
    actor: ActorSettings
    critic: VCriticSettings
