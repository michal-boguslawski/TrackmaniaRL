import torch as T
from torch import nn

from rl_lib.networks.utils import init_layer


class CNN(nn.Module):
    def __init__(self, observation_dim: int):
        super().__init__()
        self._network = nn.Sequential(
            init_layer(nn.Conv2d(observation_dim, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(4096, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 256)),
            # nn.LayerNorm(256),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self._network(x)
