"""Actor network with Beta distribution output and affine action transform.

The actor outputs Beta distribution parameters (alpha, beta) for each action
dimension, transformed via AffineTransform to the environment's action bounds.
This provides bounded continuous actions with differentiable log-prob gradients.
"""

import torch as T
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Beta, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from rl_lib.networks.config import ActorConfig
from rl_lib.networks.utils import init_layer, make_activation


class StableTanhTransform(TanhTransform):
    """TanhTransform with clamped inverse and log-det-jacobian for stability."""

    def __init__(self, epsilon: float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def log_abs_det_jacobian(self, x, y):
        y_clamped = y.clamp(-1 + self.epsilon, 1 - self.epsilon)
        return T.log(1 - y_clamped.pow(2) + self.epsilon)

    def _inverse(self, y):
        y_clamped = y.clamp(-1 + self.epsilon, 1 - self.epsilon)
        return 0.5 * (T.log1p(y_clamped) - T.log1p(-y_clamped))


class Actor(nn.Module):
    """MLP actor head producing Beta distribution parameters per action dimension.

    Architecture: configurable hidden layers -> Linear(2 * action_dim) ->
    softplus -> Beta(alpha, beta) -> AffineTransform to [action_low, action_high].

    Attributes:
        action_dim (int): Dimensionality of the action space.
        in_dim (int): Input feature dimension from temporal encoder.
        cfg (ActorConfig): Configuration for hidden layers, init, bounds.
    """

    def __init__(self, action_dim: int, in_dim: int, config: ActorConfig):
        super().__init__()
        self.action_dim = action_dim
        self.in_dim = in_dim
        self.cfg = config
        if len(self.cfg.action_low) != action_dim or len(self.cfg.action_high) != action_dim:
            raise ValueError(
                f"actor action bounds must each have action_dim={action_dim} values; "
                f"got {len(self.cfg.action_low)} and {len(self.cfg.action_high)}"
            )

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
            nn.Linear(current_dim, 2 * action_dim, bias=self.cfg.output_bias),
            self.cfg.output_init_gain,
            self.cfg.output_init_bias,
        ))
        self._network = nn.Sequential(*modules)
        low = T.tensor(self.cfg.action_low)
        high = T.tensor(self.cfg.action_high)
        self.register_buffer("_affine_loc", low)
        self.register_buffer("_affine_scale", high - low)

    @property
    def out_dim(self) -> int:
        """Output dimension (equals action_dim)."""
        return self.action_dim

    def action_transform(self, action: T.Tensor) -> T.Tensor:
        """Map action from [0, 1] (Beta support) to environment bounds.

        Args:
            action: Tensor of shape (..., action_dim) in [0, 1].

        Returns:
            Tensor of shape (..., action_dim) scaled to [action_low, action_high].
        """
        return self._affine_loc + self._affine_scale * action

    def _distribution(self, x: T.Tensor, temperature: float) -> tuple[Distribution, T.Tensor]:
        """Construct Beta distribution and compute mean action.

        Args:
            x: Input tensor of shape (batch, in_dim).
            temperature: Temperature scaling for concentration parameters.
                Higher temperature -> more concentrated (deterministic) distribution.

        Returns:
            Tuple of (TransformedDistribution, mean_action) where mean_action
            is the affine-transformed Beta mean.
        """
        raw = self._network(x)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)
        alpha = F.softplus(alpha_raw, beta=self.cfg.softplus_beta, threshold=self.cfg.softplus_threshold) / temperature
        beta = F.softplus(beta_raw, beta=self.cfg.softplus_beta, threshold=self.cfg.softplus_threshold) / temperature
        alpha = alpha + self.cfg.concentration_offset
        beta = beta + self.cfg.concentration_offset
        base_dist = Beta(alpha, beta)
        transform = AffineTransform(loc=self._affine_loc, scale=self._affine_scale)
        return TransformedDistribution(base_dist, transform), transform(base_dist.mean)

    def forward(self, x: T.Tensor, temperature: float | None = None) -> tuple[Distribution, T.Tensor]:
        """Stochastic forward pass: sample from Beta distribution.

        Args:
            x: Input tensor of shape (batch, in_dim).
            temperature: Sampling temperature. Defaults to config.default_temperature.
                Must be > 0 for stochastic evaluation.

        Returns:
            Tuple of (action_distribution, action_mean).

        Raises:
            ValueError: If temperature <= 0.
        """
        temperature = self.cfg.default_temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive for stochastic actor evaluation")
        return self._distribution(x, temperature)

    def forward_deterministic(self, x: T.Tensor) -> tuple[Distribution, T.Tensor]:
        """Deterministic forward pass: return mean of Beta distribution.

        Uses config.deterministic_distribution_temperature (typically small
        but non-zero to avoid degenerate Beta at boundaries).

        Args:
            x: Input tensor of shape (batch, in_dim).

        Returns:
            Tuple of (action_distribution, action_mean).
        """
        return self._distribution(x, self.cfg.deterministic_distribution_temperature)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        return f"{self.__class__.__name__}(action_dim={self.action_dim}, in_dim={self.in_dim}, config={self.cfg.model_dump()}, params={n_params:,})"

    def config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable hyperparameters."""
        params = list(super().parameters())
        return {
            "action_dim": self.action_dim,
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in params),
            "n_trainable_params": sum(p.numel() for p in params if p.requires_grad),
        }
