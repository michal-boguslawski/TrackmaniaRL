from logging import getLogger
import torch as T
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Beta, TransformedDistribution
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
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.GELU(),
            init_layer(nn.Linear(self.cfg.hidden_dim, 2 * action_dim), 0.01, -2.0),
        )

        # register as buffers instead of plain tensors
        self.register_buffer("_affine_loc", T.tensor([-1., 0., 0.]))
        self.register_buffer("_affine_scale", T.tensor([2., 1., 1.]))

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    def forward(self, x: T.Tensor, temperature: float = 1.) -> tuple[Distribution, T.Tensor]:
        raw = self._network(x)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)  # each (batch, action_dim)

        alpha = F.softplus(alpha_raw) / temperature + 1.0
        beta = F.softplus(beta_raw) / temperature + 1.0

        base_dist = Beta(alpha, beta)
        dist = TransformedDistribution(
            base_dist,
            AffineTransform(loc=self._affine_loc, scale=self._affine_scale),
        )
        return dist, base_dist.mean

    def forward_deterministic(self, x: T.Tensor) -> tuple[Distribution, T.Tensor]:
        raw = self._network(x)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)  # each (batch, action_dim)

        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        base_dist = Beta(alpha, beta)
        affine_transform = AffineTransform(loc=self._affine_loc, scale=self._affine_scale)
        dist = TransformedDistribution(
            base_dist,
            affine_transform,
        )
        return dist, affine_transform(base_dist.mean)

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
