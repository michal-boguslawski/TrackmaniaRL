from logging import getLogger
import torch as T
from torch import nn
from torch.distributions import Distribution, Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    TanhTransform,
)

from rl_lib.networks.utils import init_layer
from rl_lib.networks.config import ActorConfig


logger = getLogger(__name__)


class StableTanhTransform(TanhTransform):
    def log_abs_det_jacobian(self, x, y):
        # y = tanh(x); clamp away from ±1 so log(1 - y^2) can't blow up
        y_clamped = y.clamp(-1 + 1e-6, 1 - 1e-6)
        return T.log(1 - y_clamped.pow(2) + 1e-6)

    def _inverse(self, y):
        y_clamped = y.clamp(-1 + 1e-6, 1 - 1e-6)
        return 0.5 * (T.log1p(y_clamped) - T.log1p(-y_clamped))


class Actor(nn.Module):
    def __init__(self, action_dim: int, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.in_dim = in_dim

        self.cfg = ActorConfig(hidden_dim=hidden_dim)

        self._network = nn.Sequential(
            init_layer(nn.Linear(in_dim, self.cfg.hidden_dim)),
            nn.GELU(),
            init_layer(nn.Linear(self.cfg.hidden_dim, action_dim), gain=0.01),
        )
        self._log_std = nn.Parameter(T.full((action_dim,), -0.5))
        # register as buffers instead of plain tensors
        self.register_buffer("_affine_loc", T.tensor([0., 0.5, 0.5]))
        self.register_buffer("_affine_scale", T.tensor([1., 0.5, 0.5]))

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    def forward(self, x: T.Tensor, temperature: float = 1.) -> Distribution:
        mean = self._network(x).clamp(-3., 3.)
        std = self._log_std.clamp(-2.0, 0.5).exp() * temperature

        if not T.isfinite(mean).all():
            logger.error(f"mean is not finite {mean}")
            raise ValueError(f"Mean is not finite")

        if not T.isfinite(std).all():
            logger.error(f"std is not finite {std}")
            raise ValueError(f"Std is not finite")

        dist = TransformedDistribution(
            Normal(mean, std),
            ComposeTransform([
                StableTanhTransform(),
                AffineTransform(loc=self._affine_loc, scale=self._affine_scale),
            ])
        )

        return dist

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        cfg_fields = ", ".join(f"{k}={v}" for k, v in self.cfg.model_dump().items())
        return (
            f"{self.__class__.__name__}("
            f"action_dim={self.action_dim}, "
            f"in_dim={self.in_dim}, "
            f"{cfg_fields}, "
            f"params={n_params:,})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters + param count, suitable for mlflow.log_params (with a prefix)."""
        params = list(super().parameters())
        return {
            "action_dim": self.action_dim,
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in params),
            "n_trainable_params": sum(p.numel() for p in params if p.requires_grad),
        }
