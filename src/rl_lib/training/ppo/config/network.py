from __future__ import annotations

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.actor import ActorSettings
from rl_lib.networks.settings.critic import VCriticSettings
from rl_lib.networks.settings.encoder import EncoderConfig


class PPONetworkConfig(StrictModel):
    encoder: EncoderConfig
    actor: ActorSettings
    critic: VCriticSettings
