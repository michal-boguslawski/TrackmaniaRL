import torch as T
from torch import nn
from torch.distributions import Distribution, Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    TanhTransform,
)

from rl_lib.networks.utils import init_layer


class Actor(nn.Module):
    def __init__(self, action_dim: int, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self._network = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, action_dim), gain=0.01),  # near-uniform initial policy
        )
        self._log_std = nn.Parameter(T.full((action_dim,), -0.5))
        # register as buffers instead of plain tensors
        self.register_buffer("_affine_loc", T.tensor([0., 0.5, 0.5]))
        self.register_buffer("_affine_scale", T.tensor([1., 0.5, 0.5]))
        self._transforms = ComposeTransform(
            [
                TanhTransform(),
                AffineTransform(loc=self._affine_loc, scale=self._affine_scale),
            ]
        )

    def forward(self, x: T.Tensor, temperature: float = 1.) -> Distribution:
        mean = self._network(x).clamp(-6, 6)
        std = self._log_std.clamp(-2.0, 0.5).exp() * temperature

        assert T.isfinite(mean).all(), mean
        assert T.isfinite(std).all(), std

        return TransformedDistribution(
            Normal(mean, std),
            transforms=self._transforms
        )

    def act_deterministic(self, x: T.Tensor) -> T.Tensor:
        mean = self._network(x)
        action = self._transforms(mean)
        return action
