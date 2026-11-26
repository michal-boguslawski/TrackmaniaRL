"""PPO objective terms and their diagnostic metrics."""

from dataclasses import dataclass

import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.entropy import EntropyDecay
from rl_lib.training.ppo.config.loss import (
    PPOActorLossSettings,
    PPOCriticLossSettings,
    PPOLossConfig,
)
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext


@dataclass
class PPOLossOutput:
    total: T.Tensor
    policy: T.Tensor
    value: T.Tensor
    entropy: T.Tensor


class PPOLosses:
    """Evaluate PPO actor, critic, and entropy objectives.

    The settings are supplied to each calculation so replacing a trainer's
    configuration takes effect immediately rather than leaving stale settings
    in this component. The verbosity decides how much of the diagnostic
    metrics is computed: below ``Verbosity.ALL`` the reductions behind them
    are skipped, while the losses themselves are always produced.
    """

    def __init__(
        self,
        cfg: PPOLossConfig,
        verbosity: int | Verbosity = Verbosity.ALL,
    ):
        """Initialize loss evaluation for an agent.

        Args:
            agent: Agent used to evaluate actions and provide the critic Huber
                loss delta from its network configuration.
            verbosity: Metric detail level deciding which diagnostics to compute.

        Raises:
            ValueError: If verbosity is not one of 0, 1, or 2.
        """
        self.cfg = cfg
        self._verbosity = as_verbosity(verbosity)
        self._critic_loss_fn = nn.HuberLoss(
            reduction="none", delta=cfg.critic_loss_cfg.huber_delta
        )
        self.entropy_decay = EntropyDecay(cfg.entropy)

    def calculate_losses(self, update_context: PPOUpdateContext) -> PPOLossOutput:
        """Compute PPO losses while retaining metrics as device tensors."""
        actor_loss = self.actor_loss(self.cfg.actor_loss_cfg, update_context)

        critic_loss = self.critic_loss(
            self.cfg.critic_loss_cfg, self._critic_loss_fn, update_context
        )
        entropy_loss = self.entropy_loss(update_context)

        total_loss = (
            actor_loss
            + self.cfg.critic_loss_cfg.critic_beta * critic_loss
            - self.entropy_decay.coefficient * entropy_loss
        )
        return PPOLossOutput(total_loss, actor_loss, critic_loss, entropy_loss)

    @staticmethod
    def actor_loss(
        actor_loss_cfg: PPOActorLossSettings,
        update_context: PPOUpdateContext,
    ) -> T.Tensor:
        """Compute actor loss and leave diagnostic tensors on their device."""
        advantages = update_context.advantages
        if actor_loss_cfg.advantage_normalization_strategy == "batch":
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + actor_loss_cfg.advantage_epsilon
            )

        assert update_context.new_log_probs.shape == update_context.old_log_probs.shape
        log_ratio = update_context.new_log_probs - update_context.old_log_probs
        ratio = log_ratio.sum(-1).exp()
        clipped_ratio = ratio.clamp(
            min=1 - actor_loss_cfg.ppo_epsilon, max=1 + actor_loss_cfg.ppo_epsilon
        )

        assert advantages.shape == ratio.shape
        surrogate_loss = -T.minimum(
            ratio * advantages, clipped_ratio * advantages
        ).mean()

        # mean_ref = (update_context.actions**2).mean()
        # Zero represents no input for steering, gas, and brake
        actor_loss = surrogate_loss
        # + actor_loss_cfg.mean_reg_coef * mean_ref

        return actor_loss

    @staticmethod
    def critic_loss(
        critic_loss_cfg: PPOCriticLossSettings,
        critic_loss_fn: nn.HuberLoss,
        update_context: PPOUpdateContext,
    ) -> T.Tensor:
        """Compute critic loss and leave its diagnostic tensor on-device."""
        values = update_context.values
        old_values = update_context.old_values
        returns = update_context.returns
        assert values.shape == old_values.shape == returns.shape
        value_clip_epsilon = critic_loss_cfg.value_clip_epsilon
        clipped_values = (
            old_values
            + (values - old_values).clamp(-value_clip_epsilon, value_clip_epsilon)
            if value_clip_epsilon
            else values
        )
        loss_unclipped = critic_loss_fn(values, returns)
        loss_clipped = critic_loss_fn(clipped_values, returns)
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()
        return critic_loss

    @staticmethod
    def entropy_loss(update_context: PPOUpdateContext) -> T.Tensor:
        """Compute entropy and leave diagnostics on-device."""
        entropy = update_context.dist.base_dist.entropy()
        assert isinstance(entropy, T.Tensor)
        entropy_loss = entropy.sum(dim=-1).mean()
        return entropy_loss
