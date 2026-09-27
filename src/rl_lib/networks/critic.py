"""Critic network: MLP value head.

Maps temporal encoding features to a scalar state value estimate.
"""

import torch as T
from torch import nn

from rl_lib.networks.config import CriticConfig
from rl_lib.networks.utils import init_layer, make_activation


class Critic(nn.Module):
    """MLP value head producing scalar state-value estimates.

    Attributes:
        in_dim (int): Input feature dimension from temporal encoder.
        cfg (CriticConfig): Configuration for hidden layers and initialization.
    """

    def __init__(self, in_dim: int, config: CriticConfig):
        super().__init__()
        self.in_dim = in_dim
        self.cfg = config

        hidden_layers = self.cfg.hidden_layers
        modules: list[nn.Module] = []
        current_dim = in_dim
        for layer_cfg in hidden_layers:
            modules.extend((
                init_layer(
                    nn.Linear(current_dim, layer_cfg.out_dim, bias=layer_cfg.bias),
                    layer_cfg.init_gain,
                    layer_cfg.init_bias,
                ),
                make_activation(layer_cfg.activation),
            ))
            current_dim = layer_cfg.out_dim
        modules.append(init_layer(
            nn.Linear(current_dim, 1, bias=self.cfg.output_bias),
            self.cfg.output_init_gain,
            self.cfg.output_init_bias,
        ))
        self._network = nn.Sequential(*modules)

    @property
    def out_dim(self) -> int:
        """Output dimension (always 1 for scalar value)."""
        return 1

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Compute state value from temporal encoding.

        Args:
            x: Tensor of shape (batch, in_dim).

        Returns:
            Tensor of shape (batch,) scalar value per batch item.
        """
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
