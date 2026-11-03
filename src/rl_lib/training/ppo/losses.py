"""PPO objective terms and their diagnostic metrics."""

import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.settings.loss import (
    EntropySettings,
    PPOActorLossSettings,
    PPOCriticLossSettings,
    PPOLossSettings,
)


class EntropyDecay:
    def __init__(self, cfg: EntropySettings):
        self.coefficient_init = cfg.coefficient
        self.decay = cfg.decay
        self.coefficient_min = cfg.min_coefficient
        self._factor = 1.0

    @property
    def coefficient(self) -> float:
        return max(
            self.coefficient_init * self._factor,
            self.coefficient_min,
        )

    def step(self) -> None:
        self._factor *= self.decay


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
        cfg: PPOLossSettings,
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
        self._entropy_decay = EntropyDecay(cfg.entropy)

    def calculate_losses(
        self,
        log_probs: T.Tensor,
        critic_values: T.Tensor,
        dist: Distribution,
        entropy_decay: EntropyDecay,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute PPO losses while retaining metrics as device tensors."""
        actor_loss, actor_metrics = self.actor_loss(
            self.cfg.actor_loss_cfg, advantages, log_probs, old_log_probs
        )
        critic_loss, critic_metrics = self.critic_loss(
            self.cfg.critic_loss_cfg, returns, critic_values, old_values
        )
        entropy_loss, entropy_metrics = self.entropy_loss(dist)
        metrics = {**actor_metrics, **critic_metrics, **entropy_metrics}

        total_loss = (
            actor_loss
            + self.cfg.critic_loss_cfg.critic_beta * critic_loss
            - entropy_decay.coefficient * entropy_loss
        )
        metrics["loss/total"] = total_loss.detach()
        return total_loss, metrics

    @staticmethod
    def actor_loss(
        actor_loss_cfg: PPOActorLossSettings,
        advantages: T.Tensor,
        log_probs: T.Tensor,
        old_log_probs: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute actor loss and leave diagnostic tensors on their device."""
        if actor_loss_cfg.advantage_normalization_strategy == "batch":
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + actor_loss_cfg.advantage_epsilon
            )

        assert log_probs.shape == old_log_probs.shape
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.sum(-1).exp()
        clipped_ratio = ratio.clamp(
            min=1 - actor_loss_cfg.ppo_epsilon, max=1 + actor_loss_cfg.ppo_epsilon
        )

        assert advantages.shape == ratio.shape
        surrogate_loss = -T.minimum(
            ratio * advantages, clipped_ratio * advantages
        ).mean()

        # Zero represents no input for steering, gas, and brake
        actor_loss = surrogate_loss
        return actor_loss, {"loss/actor": actor_loss.detach()}

    @staticmethod
    def critic_loss(
        critic_loss_cfg: PPOCriticLossSettings,
        critic_loss_fn: nn.HuberLoss,
        returns: T.Tensor,
        values: T.Tensor,
        old_values: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute critic loss and leave its diagnostic tensor on-device."""
        assert values.shape == old_values.shape == returns.shape
        value_clip_epsilon = critic_loss_cfg.value_clip_epsilon
        clipped_values = old_values + (values - old_values).clamp(
            -value_clip_epsilon, value_clip_epsilon
        )
        loss_unclipped = critic_loss_fn(values, returns)
        loss_clipped = critic_loss_fn(clipped_values, returns)
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()
        return critic_loss, {"loss/critic": critic_loss.detach()}

    @staticmethod
    def entropy_loss(dist: Distribution) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute entropy and leave diagnostics on-device."""
        entropy: T.Tensor = dist.base_dist.entropy()
        entropy_loss = entropy.sum(dim=-1).mean()
        return entropy_loss, {"loss/entropy": entropy_loss.detach()}
