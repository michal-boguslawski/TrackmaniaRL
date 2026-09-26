import torch as T
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Beta, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from rl_lib.networks.config import ActorConfig
from rl_lib.networks.utils import init_layer, make_activation


class StableTanhTransform(TanhTransform):
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
    def __init__(self, action_dim: int, in_dim: int, hidden_dim: int | None = None, config: ActorConfig | None = None):
        super().__init__()
        self.action_dim = action_dim
        self.in_dim = in_dim
        if config is not None:
            self.cfg = config
        elif hidden_dim is not None:
            default_layer = ActorConfig().hidden_layers[0]
            self.cfg = ActorConfig(hidden_layers=[default_layer.model_copy(update={"out_dim": hidden_dim})])
        else:
            self.cfg = ActorConfig()
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
        return self.action_dim

    def action_transform(self, action: T.Tensor) -> T.Tensor:
        return self._affine_loc + self._affine_scale * action

    def _distribution(self, x: T.Tensor, temperature: float) -> tuple[Distribution, T.Tensor]:
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
        temperature = self.cfg.default_temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive for stochastic actor evaluation")
        return self._distribution(x, temperature)

    def forward_deterministic(self, x: T.Tensor) -> tuple[Distribution, T.Tensor]:
        return self._distribution(x, self.cfg.deterministic_distribution_temperature)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        return f"{self.__class__.__name__}(action_dim={self.action_dim}, in_dim={self.in_dim}, config={self.cfg.model_dump()}, params={n_params:,})"

    def config(self) -> dict[str, int | float | str]:
        params = list(super().parameters())
        return {
            "action_dim": self.action_dim,
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in params),
            "n_trainable_params": sum(p.numel() for p in params if p.requires_grad),
        }
