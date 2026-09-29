"""PPO objective terms and their diagnostic metrics."""

from torch import nn
import torch as T
from torch.distributions import Distribution

from rl_lib.agent import Agent
from rl_lib.run_config import TrainerSettings
from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.ppo.metrics import _tensor_metrics_to_scalars


class PPOLosses:
    """Evaluate PPO actor, critic, and entropy objectives.

    The settings are supplied to each calculation so replacing a trainer's
    configuration takes effect immediately rather than leaving stale settings
    in this component. The verbosity decides how much of the diagnostic
    metrics is computed: below ``Verbosity.ALL`` the reductions behind them
    are skipped, while the losses themselves are always produced.
    """

    def __init__(self, agent: Agent, verbosity: int | Verbosity = Verbosity.ALL):
        """Initialize loss evaluation for an agent.

        Args:
            agent: Agent used to evaluate actions and provide the critic Huber
                loss delta from its network configuration.
            verbosity: Metric detail level deciding which diagnostics to compute.

        Raises:
            ValueError: If verbosity is not one of 0, 1, or 2.
        """
        self._agent = agent
        self._verbosity = as_verbosity(verbosity)
        self._critic_loss_fn = nn.HuberLoss(
            reduction="none", delta=agent.network.cfg.critic.huber_delta
        )

    def calculate_losses(
        self,
        cfg: TrainerSettings,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None,
    ) -> tuple[tuple[T.Tensor, T.Tensor, T.Tensor], dict[str, float]]:
        """Evaluate policy actions and compute all PPO loss components.

        Args:
            cfg: Current trainer settings.
            advantages: GAE advantages with shape (batch,).
            returns: GAE returns with shape (batch,).
            old_log_probs: Old policy log-probabilities with shape
                (batch, action_dim).
            old_values: Old value estimates with shape (batch,).
            observation: Flattened NHWC observation windows with shape
                (batch * seq, height, width, channel).
            action: Actions with shape (batch, action_dim).
            dones: Temporal masking flags with shape (batch * seq,), or None.

        Returns:
            Tuple of actor, critic, and entropy losses, plus scalar metrics.
        """
        losses, metric_tensors = self._calculate_losses(
            cfg,
            advantages,
            returns,
            old_log_probs,
            old_values,
            observation,
            action,
            dones,
        )
        return losses, _tensor_metrics_to_scalars(metric_tensors)

    def _calculate_losses(
        self,
        cfg: TrainerSettings,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None,
    ) -> tuple[tuple[T.Tensor, T.Tensor, T.Tensor], dict[str, T.Tensor]]:
        """Compute PPO losses while retaining metrics as device tensors."""
        log_probs, values, dist, action_mean = self._agent.evaluate_actions(
            observation, action, dones
        )
        actor_loss, actor_metrics = self._actor_loss(
            cfg, advantages, log_probs, old_log_probs, action_mean
        )
        critic_loss, critic_metrics = self._critic_loss(
            cfg, returns, values, old_values
        )
        entropy_loss, entropy_metrics = self._entropy_loss(dist)
        metrics = {**actor_metrics, **critic_metrics, **entropy_metrics}
        return (actor_loss, critic_loss, entropy_loss), metrics

    def actor_loss(
        self,
        cfg: TrainerSettings,
        advantages: T.Tensor,
        log_probs: T.Tensor,
        old_log_probs: T.Tensor,
        action_mean: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, float]]:
        """Compute the clipped actor surrogate with mean-action regularization.

        Args:
            cfg: Current trainer settings.
            advantages: GAE advantages with shape (batch,).
            log_probs: New policy log-probabilities with shape
                (batch, action_dim).
            old_log_probs: Old policy log-probabilities with shape
                (batch, action_dim).
            action_mean: New policy mean actions with shape
                (batch, action_dim).

        Returns:
            Actor loss tensor and scalar diagnostics.
        """
        loss, metric_tensors = self._actor_loss(
            cfg, advantages, log_probs, old_log_probs, action_mean
        )
        return loss, _tensor_metrics_to_scalars(metric_tensors)

    def _actor_loss(
        self,
        cfg: TrainerSettings,
        advantages: T.Tensor,
        log_probs: T.Tensor,
        old_log_probs: T.Tensor,
        action_mean: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute actor loss and leave diagnostic tensors on their device."""
        if cfg.advantage_normalization_strategy == "batch":
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + cfg.advantage_epsilon
            )

        assert log_probs.shape == old_log_probs.shape
        log_ratio = (log_probs - old_log_probs).clamp(
            cfg.log_ratio_min, cfg.log_ratio_max
        )
        ratio = log_ratio.sum(-1).clamp(-5, 5).exp()
        clipped_ratio = ratio.clamp(
            min=1 - cfg.ppo_epsilon, max=1 + cfg.ppo_epsilon
        )

        assert advantages.shape == ratio.shape
        surrogate_loss = -T.minimum(
            ratio * advantages, clipped_ratio * advantages
        ).mean()

        # Zero represents no input for steering, gas, and brake.
        mean_reg = action_mean.pow(2).mean()
        actor_loss = surrogate_loss + cfg.mean_reg_coef * mean_reg
        metrics = self._actor_loss_metrics(
            cfg, log_ratio, surrogate_loss, mean_reg, action_mean, actor_loss
        )
        return actor_loss, metrics

    def _actor_loss_metrics(
        self,
        cfg: TrainerSettings,
        log_ratio: T.Tensor,
        surrogate_loss: T.Tensor,
        mean_reg: T.Tensor,
        action_mean: T.Tensor,
        actor_loss: T.Tensor,
    ) -> dict[str, T.Tensor]:
        """Compute diagnostics for the actor objective."""
        with T.no_grad():
            log_ratio_total = log_ratio.sum(-1)
            ratio_total = log_ratio_total.exp()
            # The KL is the only diagnostic the update loop reads to stop an
            # epoch early, so it is always computed.
            approx_kl = (ratio_total - 1 - log_ratio_total).mean()
            metric_tensors = {
                "metrics/actor_loss": actor_loss.detach(),
                "metrics/approx_kl": approx_kl,
            }
            if not self._verbosity.logs_diagnostics:
                return metric_tensors
            ratio_per_action = log_ratio.exp()
            approx_kl_per_action = (ratio_per_action - 1 - log_ratio).mean(dim=0)
            ratio_max = ratio_total.max()
            clip_fraction = (
                T.abs(ratio_total - 1) > cfg.ppo_epsilon
            ).float().mean()
            metric_tensors.update(
                {
                    "metrics/ratio_max": ratio_max,
                    "metrics/clip_fraction": clip_fraction,
                    "metrics/surrogate_loss": surrogate_loss.detach(),
                    "metrics/mean_reg": mean_reg.detach(),
                    "metrics/mean_abs_max": action_mean.abs().max(),
                    "metrics/tanh_saturation_frac": (
                        action_mean.abs() > cfg.tanh_saturation_threshold
                    ).float().mean(),
                }
            )
            mean_abs_max_by_action = action_mean.abs().amax(dim=0)
            metric_tensors.update({
                f"metrics/mean_abs_max_{index}": value
                for index, value in enumerate(mean_abs_max_by_action)
            })
            metric_tensors.update(
                {
                    f"metrics/approx_kl_{index}": value
                    for index, value in enumerate(approx_kl_per_action)
                }
            )
        return metric_tensors

    def critic_loss(
        self,
        cfg: TrainerSettings,
        returns: T.Tensor,
        values: T.Tensor,
        old_values: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, float]]:
        """Compute the clipped Huber value loss.

        Args:
            cfg: Current trainer settings.
            returns: GAE returns with shape (batch,).
            values: New value estimates with shape (batch,).
            old_values: Old value estimates with shape (batch,).

        Returns:
            Critic loss tensor and scalar metrics.
        """
        loss, metric_tensors = self._critic_loss(cfg, returns, values, old_values)
        return loss, _tensor_metrics_to_scalars(metric_tensors)

    def _critic_loss(
        self,
        cfg: TrainerSettings,
        returns: T.Tensor,
        values: T.Tensor,
        old_values: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute critic loss and leave its diagnostic tensor on-device."""
        assert values.shape == old_values.shape == returns.shape
        value_clip_epsilon = cfg.value_clip_epsilon or cfg.ppo_epsilon
        clipped_values = old_values + (values - old_values).clamp(
            -value_clip_epsilon, value_clip_epsilon
        )
        loss_unclipped = self._critic_loss_fn(values, returns)
        loss_clipped = self._critic_loss_fn(clipped_values, returns)
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()
        return critic_loss, {"loss/critic": critic_loss.detach()}

    def entropy_loss(
        self, dist: Distribution
    ) -> tuple[T.Tensor, dict[str, float]]:
        """Compute the base distribution entropy bonus and diagnostics.

        Args:
            dist: Actor's transformed distribution. Entropy is computed from
                its base distribution because the transform has no closed form.

        Returns:
            Positive entropy bonus tensor and scalar metrics.
        """
        entropy_loss, metric_tensors = self._entropy_loss(dist)
        return entropy_loss, _tensor_metrics_to_scalars(metric_tensors)

    def _entropy_loss(
        self, dist: Distribution
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute entropy and leave diagnostics on-device."""
        entropy: T.Tensor = dist.base_dist.entropy()
        entropy_loss = entropy.sum(dim=-1).mean()
        metric_tensors = {"loss/entropy": entropy_loss.detach()}
        if self._verbosity.logs_diagnostics:
            with T.no_grad():
                log_std = dist.base_dist.variance.pow(1 / 2).log().mean(0)
                metric_tensors.update(
                    {
                        f"metrics/entropy_{index}": value
                        for index, value in enumerate(entropy.mean(0))
                    }
                )
                metric_tensors.update(
                    {
                        f"metrics/log_std_{index}": value
                        for index, value in enumerate(log_std)
                    }
                )
        return entropy_loss, metric_tensors
