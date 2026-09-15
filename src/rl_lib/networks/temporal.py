import torch as T
from torch import nn

from rl_lib.networks.config import TemporalConfig


class TemporalCNN1D(nn.Module):
    def __init__(self, stack_size: int, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.stack_size = stack_size
        self.in_dim = in_dim

        self.cfg = TemporalConfig(out_dim=out_dim)

        self._encoder = nn.Conv1d(in_dim, self.cfg.out_dim, kernel_size=stack_size)
        self._norm = nn.LayerNorm(in_dim)
        self._out_norm = nn.LayerNorm(self.cfg.out_dim)

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self._norm(x)
        x = x.permute(0, 2, 1)
        x = self._encoder(x)
        x = x.permute(0, 2, 1)
        x.squeeze_(1)
        x = self._out_norm(x)
        return x

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        cfg_fields = ", ".join(f"{k}={v}" for k, v in self.cfg.model_dump().items())
        return (
            f"{self.__class__.__name__}("
            f"stack_size={self.stack_size}, "
            f"in_dim={self.in_dim}, "
            f"{cfg_fields}, "
            f"params={n_params:,})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters + param count, suitable for mlflow.log_params (with a prefix)."""
        return {
            "stack_size": self.stack_size,
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in super().parameters()),
        }
