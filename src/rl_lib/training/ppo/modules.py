import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import VCritic
from rl_lib.networks.encoder import Encoder


class ActorCritic(nn.Module):
    def __init__(self, encoder: Encoder, actor: Actor, critic: VCritic):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

    def forward(self, x: T.Tensor, mask: T.Tensor) -> tuple[Distribution, T.Tensor]:
        features = self.encoder(
            x, mask
        )  # (B,T,H,W,C) -> (B,D), the training entry point
        action_dist, _ = self.actor(features)
        critic_value = self.critic(features)
        return action_dist, critic_value
