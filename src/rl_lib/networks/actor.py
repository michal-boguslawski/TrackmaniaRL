import torch as T
from torch import nn
from torch.distributions import Distribution, Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    TanhTransform,
)

from src.rl_lib.networks.utils import init_layer


class Actor(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self._network = nn.Sequential(
            init_layer(nn.Linear(256, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, action_dim), gain=0.01),  # near-uniform initial policy
        )
        self._log_std = nn.Parameter(T.full((action_dim,), -0.5))

        # register as buffers instead of plain tensors
        self.register_buffer("_affine_loc", T.tensor([0., 0.5, 0.5]))
        self.register_buffer("_affine_scale", T.tensor([1., 0.5, 0.5]))

    def forward(self, x: T.Tensor, temperature: float = 1.) -> Distribution:
        mean = self._network(x)
        std = self._log_std.clamp(-2.0, 0.5).exp() * temperature

        assert T.isfinite(mean).all(), mean
        assert T.isfinite(std).all(), std

        return TransformedDistribution(
            Normal(mean, std),
            transforms=ComposeTransform(
                [
                    TanhTransform(),
                    AffineTransform(loc=self._affine_loc, scale=self._affine_scale),
                ]
            )
        )


if __name__ == "__main__":
    actor = Actor(3)
    test = T.randn(10, 512)
    dist = actor(test)
    action = dist.sample()
    print(action.shape)
    print(dist.log_prob(action).shape)
    print(dist.base_dist.entropy().shape)
    print(actor)
