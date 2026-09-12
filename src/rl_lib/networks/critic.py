import torch as T
from torch import nn

from rl_lib.networks.utils import init_layer
from rl_lib.networks.config import CriticConfig


class Critic(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim

        self.cfg = CriticConfig(hidden_dim=hidden_dim)

        self._network = nn.Sequential(
            init_layer(nn.Linear(in_dim, self.cfg.hidden_dim)),
            nn.GELU(),
            init_layer(nn.Linear(self.cfg.hidden_dim, 1)),  # standard, not sqrt(2), for value head
        )

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self._network(x).squeeze_(-1)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        cfg_fields = ", ".join(f"{k}={v}" for k, v in self.cfg.model_dump().items())
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"{cfg_fields}, "
            f"params={n_params:,})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters + param count, suitable for mlflow.log_params (with a prefix)."""
        return {
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in super().parameters()),
        }
