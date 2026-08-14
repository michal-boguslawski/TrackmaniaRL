import torch as T
from torch import nn
from torch.distributions import Distribution

from src.rl_lib.networks.actor import Actor
from src.rl_lib.networks.critic import Critic
from src.rl_lib.networks.cnn import CNN


class Network(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, stack_size: int):
        super().__init__()

        self.cnn = CNN(observation_dim)
        self.sequence_encoder = nn.Conv1d(
            512, 512, kernel_size=stack_size
        )
        # self.norm = nn.LayerNorm(512)
        self.actor = Actor(action_dim)
        self.critic = Critic()

    def forward(self, observation: T.Tensor, temperature: float = 1.) -> tuple[Distribution, T.Tensor]:
        x = observation.view(-1, *observation.shape[-3:])
        x = self.cnn(x)

        x = x.view(observation.shape[0], -1, x.shape[-1])
        x = self.sequence_encoder(x.permute(0, 2, 1)).squeeze_(-1)
        # x = self.norm(x)

        action_dist = self.actor(x, temperature=temperature)
        value = self.critic(x)

        return action_dist, value
