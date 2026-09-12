import torch as T
from torch import nn

from rl_lib.networks.utils import init_layer
from rl_lib.networks.config import CNNConfig


class CNN(nn.Module):
    def __init__(self, observation_dim: int, channels: int = 16, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.observation_dim = observation_dim
        self.cfg = CNNConfig(channels=channels, hidden_dim=hidden_dim, out_dim=out_dim)
        self._network = nn.Sequential(
            init_layer(nn.Conv2d(observation_dim, self.cfg.channels, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            init_layer(nn.Conv2d(self.cfg.channels, self.cfg.channels * 2, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            init_layer(nn.Conv2d(self.cfg.channels * 2, self.cfg.channels * 4, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            init_layer(nn.Conv2d(self.cfg.channels * 4, self.cfg.channels * 4, kernel_size=3, stride=2)),  # 8x8 -> 3x3
            nn.ReLU(inplace=True),
            nn.Flatten(),
            init_layer(nn.Linear(self.cfg.channels * 4 * 3 * 3, self.cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            init_layer(nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim)),
        )

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self._network(x)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        cfg_fields = ", ".join(f"{k}={v}" for k, v in self.cfg.model_dump().items())
        return (
            f"{self.__class__.__name__}("
            f"observation_dim={self.observation_dim}, "
            f"{cfg_fields}, "
            f"params={n_params:,})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters + param count, suitable for mlflow.log_params (with a prefix)."""
        return {
            "observation_dim": self.observation_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in super().parameters()),
        }
