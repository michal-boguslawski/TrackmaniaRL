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
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        stack_size: int,
        config: NetworkConfig | None = None,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ):
        super().__init__()
        self.cfg = config or NetworkConfig()

        self.cnn = CNN(observation_dim, config=self.cfg.cnn)
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
        """Expects input of shape (batch, channel, height, width)"""
        return self.cnn(x)

    def temporal_encode(self, x: T.Tensor) -> T.Tensor:
        """Expects input of shape (batch, seq, feature)"""
        return self.sequence_encoder(x)

    def heads(self, x: T.Tensor, temperature: float) -> tuple[Distribution, T.Tensor, T.Tensor]:
        """Expects input of shape (batch, feature)"""
        if temperature == 0.:
            action_dist, action_mean = self.actor.forward_deterministic(x)
        else:
            action_dist, action_mean = self.actor(x, temperature)
        value = self.critic(x)

        return action_dist, action_mean, value

    def action_transform(self, action: T.Tensor) -> T.Tensor:
        return self.actor.action_transform(action)

    def save_state_dict(self, path: str) -> None:
        T.save(self.state_dict(), path)

    def load_state_dict(self, path: str) -> None:
        super().load_state_dict(T.load(path, weights_only=True))

    def _submodules(self) -> list[tuple[str, nn.Module]]:
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
        and param counts are reported separately from the pydantic config."""
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
        one module's output dim without updating the next module's input)."""
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
