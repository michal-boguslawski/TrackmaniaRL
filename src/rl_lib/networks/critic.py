import torch as T
from torch import nn

from src.rl_lib.networks.utils import init_layer


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self._network = nn.Sequential(
            init_layer(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 1), gain=1.0),  # standard, not sqrt(2), for value head
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self._network(x).squeeze_(-1)
