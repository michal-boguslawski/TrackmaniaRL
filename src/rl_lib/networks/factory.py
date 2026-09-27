"""Network factory: composes CNN -> TemporalCNN1D -> Actor/Critic into a single module.

The Network class is the top-level policy/value network used by Agent and PPOTrainer.
It handles the full forward pass from raw NCHW observations to action distributions
and value estimates, with shape validation via describe_shapes().
"""

import torch as T
import json
from torch import nn
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import Critic
from rl_lib.networks.cnn import CNN
from rl_lib.networks.temporal import TemporalCNN1D
from rl_lib.networks.config import NetworkConfig


class Network(nn.Module):
    """Composed policy/value network: CNN -> Temporal -> Actor + Critic.

    Flow:
        observation (NCHW) -> CNN -> (batch, cnn_out_dim)
        -> TemporalCNN1D (with stack_size) -> (batch, temporal_out_dim)
        -> Actor (Beta dist + affine) / Critic (scalar value)

    Attributes:
        cfg (NetworkConfig): Full network configuration.
        stack_size (int): Number of frames in temporal window.
        observation_dim (int): Input channels to CNN.
        action_dim (int): Action space dimensionality.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        stack_size: int,
        config: NetworkConfig,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ):
        """Initialize the composed network.

        Args:
            observation_dim: Number of input channels (e.g., 1 for grayscale).
            action_dim: Dimensionality of action space.
            stack_size: Number of frames in temporal sequence.
            config: NetworkConfig containing CNN, temporal, actor, critic configs.
            action_low: Optional override for actor action_low.
            action_high: Optional override for actor action_high.

        Raises:
            ValueError: If action_low provided without action_high or vice versa.
        """
        super().__init__()
        self.cfg = config
        self.stack_size = stack_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.cnn = CNN(observation_dim, self.cfg.cnn)

        self.sequence_encoder = TemporalCNN1D(
            stack_size, in_dim=self.cnn.out_dim, config=self.cfg.temporal
        )
        actor_config = self.cfg.actor
        if action_low is not None or action_high is not None:
            if action_low is None or action_high is None:
                raise ValueError("action_low and action_high must be provided together")
            actor_config = actor_config.model_copy(update={"action_low": action_low, "action_high": action_high})
        self.actor = Actor(
            action_dim, in_dim=self.sequence_encoder.out_dim, config=actor_config
        )
        self.critic = Critic(
            in_dim=self.sequence_encoder.out_dim, config=self.cfg.critic
        )

    def feature_extract(self, x: T.Tensor) -> T.Tensor:
        """Extract CNN features from NCHW observation.

        Args:
            x: Tensor of shape (batch, channel, height, width).

        Returns:
            Tensor of shape (batch, cnn_out_dim).
        """
        return self.cnn(x)

    def temporal_encode(self, x: T.Tensor) -> T.Tensor:
        """Encode temporal sequence of CNN features.

        Args:
            x: Tensor of shape (batch, stack_size, cnn_out_dim).

        Returns:
            Tensor of shape (batch, temporal_out_dim).
        """
        return self.sequence_encoder(x)

    def heads(self, x: T.Tensor, temperature: float) -> tuple[Distribution, T.Tensor, T.Tensor]:
        """Compute action distribution, mean action, and value from temporal encoding.

        Args:
            x: Tensor of shape (batch, temporal_out_dim).
            temperature: Sampling temperature for actor.

        Returns:
            Tuple of (action_distribution, action_mean, value) where:
            - action_distribution: TransformedDistribution (Beta + AffineTransform)
            - action_mean: Tensor of shape (batch, action_dim)
            - value: Tensor of shape (batch,)
        """
        if temperature == 0.:
            action_dist, action_mean = self.actor.forward_deterministic(x)
        else:
            action_dist, action_mean = self.actor(x, temperature)
        value = self.critic(x)

        return action_dist, action_mean, value

    def action_transform(self, action: T.Tensor) -> T.Tensor:
        """Apply actor's affine transform from [0,1] to environment bounds.

        Args:
            action: Tensor in [0, 1] range.

        Returns:
            Tensor scaled to [action_low, action_high].
        """
        return self.actor.action_transform(action)

    def save_state_dict(self, path: str) -> None:
        """Save model state dict to disk."""
        T.save(self.state_dict(), path)

    def load_state_dict(self, path: str) -> None:
        """Load model state dict from disk (weights_only=True for security)."""
        super().load_state_dict(T.load(path, weights_only=True))

    def _submodules(self) -> list[tuple[str, nn.Module]]:
        """Return named submodules for repr."""
        return [
            ("cnn", self.cnn),
            ("sequence_encoder", self.sequence_encoder),
            ("actor", self.actor),
            ("critic", self.critic),
        ]

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        submodules = "\n".join(
            f"  ({name}): {module!r}"
            for name, module in self._submodules()
        )
        return f"{self.__class__.__name__}(\n{submodules}\n  total_params={n_params:,}\n)"

    def config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable hyperparameters. Structural (env-derived) dims
        and param counts are reported separately from the pydantic config.
        """
        merged = {
            f"{prefix}.{k}": (v if isinstance(v, (int, float, str, bool)) or v is None else json.dumps(v, sort_keys=True))
            for prefix, sub in self.cfg.model_dump().items()
            for k, v in sub.items()
        }
        merged["total_params"] = sum(p.numel() for p in self.parameters())
        merged["total_trainable_params"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return merged

    def describe_shapes(self) -> str:
        """Human-readable dimension flow through the network — useful for
        catching config mismatches (e.g. a YAML edit that silently changes
        one module's output dim without updating the next module's input).
        """
        lines = [
            f"observation_dim={self.cnn.observation_dim}"
            f"  --[CNN]-->  cnn.out_dim={self.cnn.cfg.out_dim}",

            f"cnn.out_dim={self.cnn.cfg.out_dim}"
            f"  --[TemporalCNN1D, stack_size={self.sequence_encoder.stack_size}]-->  "
            f"sequence_encoder.out_dim={self.sequence_encoder.cfg.out_dim}",

            f"sequence_encoder.out_dim={self.sequence_encoder.cfg.out_dim}"
            f"  --[Actor]-->  action_dim={self.actor.action_dim}",

            f"sequence_encoder.out_dim={self.sequence_encoder.cfg.out_dim}"
            f"  --[Critic]-->  value_dim=1",
        ]

        # Flag actual mismatches rather than just describing the happy path
        mismatches = []
        if self.cnn.cfg.out_dim != self.sequence_encoder.in_dim:
            mismatches.append(
                f"cnn.out_dim={self.cnn.cfg.out_dim} != "
                f"sequence_encoder.in_dim={self.sequence_encoder.in_dim}"
            )
        if self.sequence_encoder.cfg.out_dim != self.actor.in_dim:
            mismatches.append(
                f"sequence_encoder.out_dim={self.sequence_encoder.cfg.out_dim} != "
                f"actor.in_dim={self.actor.in_dim}"
            )
        if self.sequence_encoder.cfg.out_dim != self.critic.in_dim:
            mismatches.append(
                f"sequence_encoder.out_dim={self.sequence_encoder.cfg.out_dim} != "
                f"critic.in_dim={self.critic.in_dim}"
            )

        report = "\n".join(lines)
        if mismatches:
            report += "\n\n⚠ SHAPE MISMATCHES DETECTED:\n" + "\n".join(f"  - {m}" for m in mismatches)
        return report
