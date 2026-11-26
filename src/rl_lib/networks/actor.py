"""Actor network with Beta distribution output and affine action transform.

The actor outputs Beta distribution parameters (alpha, beta) for each action
dimension, transformed via AffineTransform to the environment's action bounds.
This provides bounded continuous actions with differentiable log-prob gradients.
"""

from logging import getLogger

import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, Distribution, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from rl_lib.networks.settings.actor import ActorSettings
from rl_lib.networks.utils import init_layer, make_activation, split_parameter_groups

logger = getLogger(__name__)


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


class _StableAffineTransform(AffineTransform):
    """Keep rounded affine-boundary actions inside the base distribution support."""

    def _inverse(self, y: T.Tensor) -> T.Tensor:
        """Clamp inverse values to avoid Beta log-probs of negative infinity."""
        unit = super()._inverse(y)
        epsilon = max(1e-6, T.finfo(unit.dtype).eps)
        return unit.clamp(min=epsilon, max=1.0 - epsilon)


class Actor(nn.Module):
    """MLP actor head producing Beta distribution parameters per action dimension.

    Architecture: configurable hidden layers -> Linear(2 * action_dim) ->
    softplus -> Beta(alpha, beta) -> AffineTransform to [action_low, action_high].

    Attributes:
        action_dim (int): Dimensionality of the action space.
        in_dim (int): Input feature dimension from temporal encoder.
        cfg (ActorConfig): Configuration for hidden layers, init, bounds.
    """

    def __init__(
        self,
        action_dim: int,
        in_dim: int,
        settings: ActorSettings,
        action_low: list[float] | np.ndarray | None = None,
        action_high: list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.in_dim = in_dim
        self.cfg = settings
        if len(action_low) != action_dim or len(action_high) != action_dim:
            raise ValueError(
                f"actor action bounds must each have action_dim={action_dim} values; "
                f"got {len(action_low)} and {len(action_high)}"
            )
        if (
            isinstance(self.cfg.concentration_offset, list)
            and len(self.cfg.concentration_offset) != action_dim
        ):
            raise ValueError("Wrong dim of concentration_offset")

        hidden_layers = self.cfg.hidden_layers
        modules: list[nn.Module] = []
        current_dim = in_dim
        for layer_cfg in hidden_layers:
            modules.extend(
                (
                    init_layer(
                        nn.Linear(current_dim, layer_cfg.out_dim, bias=layer_cfg.bias),
                        layer_cfg.init_gain,
                        layer_cfg.init_bias,
                    ),
                    make_activation(layer_cfg.activation),
                )
            )
            current_dim = layer_cfg.out_dim
        modules.append(
            init_layer(
                nn.Linear(current_dim, 2 * action_dim, bias=self.cfg.output_bias),
                self.cfg.output_init_gain,
                self.cfg.output_init_bias,
            )
        )
        self._network = nn.Sequential(*modules)
        low = T.tensor(action_low)
        high = T.tensor(action_high)
        self.register_buffer("_affine_loc", low)
        self.register_buffer("_affine_scale", high - low)
        self.register_buffer(
            "_concentration_offet", T.tensor(self.cfg.concentration_offset)
        )
        logger.debug("Initialized %s", self)

    @property
    def out_dim(self) -> int:
        """Output dimension (equals action_dim)."""
        return self.action_dim

    def _distribution(
        self, alpha: T.Tensor, beta: T.Tensor
    ) -> tuple[Distribution, T.Tensor]:
        """Construct Beta distribution and compute mean action.

        Args:
            x: Input tensor of shape (batch, in_dim).
            temperature: Temperature scaling for concentration parameters.
                Higher temperature -> more concentrated (deterministic) distribution.

        Returns:
            Tuple of (TransformedDistribution, mean_action) where mean_action
            is the affine-transformed Beta mean.
        """
        base_dist = Beta(alpha, beta)
        # A Beta sample can round to 0/1 after the affine map in float32. Clamp
        # the inverse in that case so replayed log-probs stay finite.
        transform = _StableAffineTransform(
            loc=self._affine_loc, scale=self._affine_scale
        )
        return TransformedDistribution(base_dist, transform), transform(base_dist.mean)

    def forward(
        self, x: T.Tensor, temperature: float | None = None
    ) -> tuple[Distribution, T.Tensor]:
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
        temperature = temperature or 1
        raw = self._network(x)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)
        alpha = (
            F.softplus(
                alpha_raw,
                beta=self.cfg.softplus_beta,
                threshold=self.cfg.softplus_threshold,
            )
            / temperature
        )
        beta = (
            F.softplus(
                beta_raw,
                beta=self.cfg.softplus_beta,
                threshold=self.cfg.softplus_threshold,
            )
            / temperature
        )
        alpha = alpha + self._concentration_offet
        beta = beta + self._concentration_offet
        dist, action_mean = self._distribution(alpha, beta)
        if temperature <= 0:
            raise ValueError(
                "temperature must be positive for stochastic actor evaluation"
            )
        return dist, action_mean

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        return (
            f"{self.__class__.__name__}("
            f"action_dim={self.action_dim}, "
            f"in_dim={self.in_dim}, "
            f"config={self.cfg.model_dump()}, "
            f"params={n_params:,})"
        )

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

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return split_parameter_groups(self)
