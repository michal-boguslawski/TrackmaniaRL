import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import Critic
from rl_lib.networks.cnn import CNN


class Network(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, stack_size: int):
        super().__init__()

        self.cnn = CNN(observation_dim)
        self.sequence_encoder = nn.Conv1d(
            256, 256, kernel_size=stack_size
        )
        self.norm = nn.LayerNorm(256)
        self.actor = Actor(action_dim)
        self.critic = Critic()

    def feature_extract(self, x: T.Tensor) -> T.Tensor:
        """Expects input of shape (batch, channel, height, width)"""
        return self.cnn(x)

    def temporal_encode(self, x: T.Tensor) -> T.Tensor:
        """Expects input of shape (batch, seq, feature)"""
        x = x.permute(0, 2, 1)
        x = self.sequence_encoder(x)
        x = x.permute(0, 2, 1)
        x.squeeze_(1)
        x = self.norm(x)
        return x

    def heads(self, x: T.Tensor, temperature: float) -> tuple[Distribution, T.Tensor]:
        """Expects input of shape (batch, feature)"""
        action_dist = self.actor(x, temperature)
        value = self.critic(x)

        return action_dist, value

    # def forward(self, observation: T.Tensor, temperature: float = 1.) -> tuple[Distribution, T.Tensor]:
    #     x = observation.view(-1, *observation.shape[-3:])
    #     x = self.cnn(x)

    #     x = x.view(observation.shape[0], -1, x.shape[-1])
    #     x = self.sequence_encoder(x.permute(0, 2, 1)).squeeze_(-1)
    #     x = self.norm(x)

    #     action_dist = self.actor(x, temperature=temperature)
    #     value = self.critic(x)

    #     return action_dist, value

    def save_state_dict(self, path: str) -> None:
        T.save(self.state_dict(), path)

    def load_state_dict(self, path: str) -> None:
        self.load_state_dict(T.load(path, weights_only=True))
