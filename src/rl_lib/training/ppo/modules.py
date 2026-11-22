import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import VCritic
from rl_lib.networks.encoder import Encoder


class ActorCritic(nn.Module):
    def __init__(self, encoder: Encoder, actor: Actor, critic: VCritic):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

    def forward(self, x: T.Tensor, mask: T.Tensor) -> tuple[Distribution, T.Tensor]:
        features = self.encoder(
            x, mask
        )  # (B,T,H,W,C) -> (B,D), the training entry point
        action_dist, _ = self.actor(features)
        critic_value = self.critic(features)
        return action_dist, critic_value

    def optimizer_parameter_groups(
        self,
        cnn_lr: float,
        sequence_encoder_lr: float,
        actor_lr: float,
        crtic_lr: float,
        weight_decay: float,
        no_decay_weight_decay: float,
    ) -> list[dict]:
        groups = []
        encoder_param_groups = self.encoder.parameter_groups()
        decay_weights = (
            ("decay", weight_decay),
            ("no_decay", no_decay_weight_decay),
        )
        components = (
            ("cnn", encoder_param_groups["cnn"], cnn_lr),
            ("temporal", encoder_param_groups["temporal"], sequence_encoder_lr),
            ("actor", self.actor.parameter_groups(), actor_lr),
            ("critic", self.critic.parameter_groups(), crtic_lr),
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
