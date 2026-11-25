"""PPO actor-critic module and optimizer parameter-group assembly."""

from typing import Any

import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import VCritic
from rl_lib.networks.encoder import Encoder
from rl_lib.training.ppo.config.learning_rates import PPOLearningRateSettings


class ActorCritic(nn.Module):
    """Shared-encoder policy and value network with grouped parameters.

    Attributes:
        encoder: Shared observation encoder exposing CNN and temporal groups.
        actor: Policy head.
        critic: Value head.
    """

    def __init__(self, encoder: Encoder, actor: Actor, critic: VCritic):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

    def forward(
        self, x: T.Tensor, mask: T.Tensor
    ) -> tuple[Distribution, T.Tensor, T.Tensor]:
        features = self.encoder(
            x, mask
        )  # (B,T,H,W,C) -> (B,D), the training entry point
        action_dist, action_mean = self.actor(features)
        critic_value = self.critic(features)
        return action_dist, critic_value, action_mean

    def optimizer_parameter_groups(
        self,
        learning_rates: PPOLearningRateSettings,
        weight_decay: float,
        no_decay_weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Build optimizer groups from a single optimizer settings object.

        Args:
            settings: Optimizer settings carrying the PPO learning rates and
                weight-decay values for decay and no-decay groups.

        Returns:
            Parameter-group dictionaries with ``params``, ``lr``,
            ``weight_decay``, and ``name`` entries.
        """
        groups: list[dict[str, Any]] = []
        encoder_param_groups = self.encoder.parameter_groups()
        decay_weights = (
            ("decay", weight_decay),
            ("no_decay", no_decay_weight_decay),
        )
        components = (
            ("cnn", encoder_param_groups["cnn"], learning_rates.cnn_lr),
            (
                "temporal",
                encoder_param_groups["temporal"],
                learning_rates.sequence_encoder_lr,
            ),
            ("actor", self.actor.parameter_groups(), learning_rates.actor_lr),
            ("critic", self.critic.parameter_groups(), learning_rates.critic_lr),
        )

        for name, param_groups, lr in components:
            for decay_type, weight_decay_value in decay_weights:
                if decay_type in param_groups:
                    groups.append(
                        {
                            "params": param_groups[decay_type],
                            "lr": lr,
                            "weight_decay": weight_decay_value,
                            "name": f"{name}_{decay_type}",
                        }
                    )

        return groups
