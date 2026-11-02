"""PPO objective terms and their diagnostic metrics."""

import torch as T
from torch import nn
from torch.distributions import Distribution

from rl_lib.policy.masks import episode_mask
from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.ppo.metrics import _tensor_metrics_to_scalars
from rl_lib.training.ppo.modules import ActorCritic
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
        modules: ActorCritic,
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
        self.modules = modules
        self._verbosity = as_verbosity(verbosity)
        self._critic_loss_fn = nn.HuberLoss(
            reduction="none", delta=cfg.critic_loss_cfg.huber_delta
        )
        self._entropy_decay = EntropyDecay(cfg.entropy)

    def calculate_losses(
        self,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None,
    ) -> tuple[T.Tensor, dict[str, float]]:
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
            self.cfg,
            self.modules,
            self._entropy_decay,
            advantages,
            returns,
            old_log_probs,
            old_values,
            observation,
            action,
            dones,
        )
        return losses, _tensor_metrics_to_scalars(metric_tensors)

    @staticmethod
    def _calculate_losses(
        cfg: PPOLossSettings,
        modules: ActorCritic,
        entropy_decay: EntropyDecay,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute PPO losses while retaining metrics as device tensors."""
        dist, critic_values = modules(observation, episode_mask(dones))
        log_probs = dist.log_prob(action)
        actor_loss, actor_metrics = PPOLosses._actor_loss(
            cfg.actor_loss_cfg, advantages, log_probs, old_log_probs
        )
        critic_loss, critic_metrics = PPOLosses._critic_loss(
            cfg.critic_loss_cfg, returns, critic_values, old_values
        )
        entropy_loss, entropy_metrics = PPOLosses._entropy_loss(dist)
        metrics = {**actor_metrics, **critic_metrics, **entropy_metrics}

        total_loss = (
            actor_loss
            + cfg.critic_loss_cfg.critic_beta * critic_loss
            - entropy_decay.coefficient * entropy_loss
        )
        metrics["loss/total"] = total_loss.detach()
        return total_loss, metrics

    def actor_loss(
        self,
        actor_loss_cfg: PPOActorLossSettings,
        advantages: T.Tensor,
        log_probs: T.Tensor,
        old_log_probs: T.Tensor,
        action_mean: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, float]]:
        """Compute the clipped actor surrogate with mean-action regularization.

        Args:
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
            actor_loss_cfg, advantages, log_probs, old_log_probs, action_mean
        )
        return loss, _tensor_metrics_to_scalars(metric_tensors)

    @staticmethod
    def _actor_loss(
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
        metrics = PPOLosses._actor_loss_metrics(
            actor_loss_cfg, log_ratio, surrogate_loss, actor_loss
        )
        return actor_loss, metrics

    def _actor_loss_metrics(
        self,
        actor_loss_cfg: PPOActorLossSettings,
        log_ratio: T.Tensor,
        surrogate_loss: T.Tensor,
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
                "loss/actor": actor_loss.detach(),
                "metrics/approx_kl": approx_kl,
            }
            if not self._verbosity.logs_diagnostics:
                return metric_tensors
            ratio_per_action = log_ratio.exp()
            approx_kl_per_action = (ratio_per_action - 1 - log_ratio).mean(dim=0)
            ratio_max = ratio_total.max()
            clip_fraction = (
                (T.abs(ratio_total - 1) > actor_loss_cfg.ppo_epsilon).float().mean()
            )
            metric_tensors.update(
                {
                    "metrics/ratio_max": ratio_max,
                    "metrics/clip_fraction": clip_fraction,
                    "metrics/surrogate_loss": surrogate_loss.detach(),
                }
            )
            metric_tensors.update(
                {
                    f"metrics/approx_kl_{index}": value
                    for index, value in enumerate(approx_kl_per_action)
                }
            )
        return metric_tensors

    def critic_loss(
        self,
        critic_loss_cfg: PPOCriticLossSettings,
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
        loss, metric_tensors = self._critic_loss(
            critic_loss_cfg, returns, values, old_values
        )
        return loss, _tensor_metrics_to_scalars(metric_tensors)

    def _critic_loss(
        self,
        critic_loss_cfg: PPOCriticLossSettings,
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
        loss_unclipped = self._critic_loss_fn(values, returns)
        loss_clipped = self._critic_loss_fn(clipped_values, returns)
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()
        return critic_loss, {"loss/critic": critic_loss.detach()}

    def entropy_loss(self, dist: Distribution) -> tuple[T.Tensor, dict[str, float]]:
        """Compute the base distribution entropy bonus and diagnostics.

        Args:
            dist: Actor's transformed distribution. Entropy is computed from
                its base distribution because the transform has no closed form.

        Returns:
            Positive entropy bonus tensor and scalar metrics.
        """
        entropy_loss, metric_tensors = self._entropy_loss(
            dist, self.cfg.logs_diagnostics
        )
        return entropy_loss, _tensor_metrics_to_scalars(metric_tensors)

    @staticmethod
    def _entropy_loss(
        dist: Distribution, logs_diagnostics: bool
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Compute entropy and leave diagnostics on-device."""
        entropy: T.Tensor = dist.base_dist.entropy()
        entropy_loss = entropy.sum(dim=-1).mean()
        metric_tensors = {"loss/entropy": entropy_loss.detach()}
        if logs_diagnostics:
            metric_tensors.update(PPOLosses._entropy_loss_metrics(entropy, dist))
        return entropy_loss, metric_tensors

    @staticmethod
    def _entropy_loss_metrics(
        entropy: T.Tensor, dist: Distribution
    ) -> dict[str, T.Tensor]:
        metric_tensors = {}
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
            metric_tensors.update(PPOLosses._beta_concentration_metrics(dist))
        return metric_tensors

    @staticmethod
    def _beta_concentration_metrics(dist: Distribution) -> dict[str, T.Tensor]:
        """Build diagnostics describing the actor's Beta concentrations.

        The actor parameterizes every action dimension with ``Beta(alpha, beta)``,
        so ``alpha`` and ``beta`` are its concentration parameters, not its mean
        and variance. They set how peaked each action distribution is, which
        makes them the first thing to inspect when the policy collapses or when
        log-probabilities turn non-finite.

        The metrics, each reduced over the minibatch:

        - ``metrics/alpha``, ``metrics/beta``: average concentration over all
          action dimensions. A steady rise means the policy is becoming
          deterministic; read it against the ``metrics/entropy_*`` decay and any
          ``metrics/approx_kl`` spikes.
        - ``metrics/alpha_{i}``, ``metrics/beta_{i}``: the same averages split
          per action dimension, which localizes the drift to steering, gas, or
          brake. Both moving together points at an unstable update, and should
          line up with ``metrics/ratio_max`` and ``metrics/approx_kl``.
        - ``metrics/concentration_max``: the largest concentration anywhere in
          the batch. A large gap to the averages marks the handful of states
          that are extremely confident, which is where non-finite
          log-probabilities originate.
        - ``metrics/beta_mean_{i}``: mean of the untransformed Beta, which lives
          in the ``[0, 1]`` support. A value near 0 or 1 is the distribution
          saturating against a support boundary, pinning the action at its
          extreme such as full brake or full gas.

        ``metrics/log_std_{i}`` is retained for compatibility with Gaussian
        policies, but it is a poor dispersion proxy here: the Beta support is
        fixed and its width is governed by the concentrations above.

        Args:
            dist: Actor's transformed distribution, read through ``base_dist``.

        Returns:
            Mapping of metric names to 0-dim or (action_dim,) tensors.
        """
        base_dist = dist.base_dist
        alpha_by_action = base_dist.concentration1.mean(0)
        beta_by_action = base_dist.concentration0.mean(0)
        # Every action dimension carries the same number of samples, so the mean
        # of the per-dimension means is the batch mean, and reusing them keeps
        # the overall average off the critical path.
        metric_tensors = {
            "metrics/alpha": alpha_by_action.mean(),
            "metrics/beta": beta_by_action.mean(),
            # Two reductions rather than an elementwise maximum over a temporary.
            "metrics/concentration_max": base_dist.concentration1.max().maximum(
                base_dist.concentration0.max()
            ),
        }
        metric_tensors.update(
            {
                f"metrics/alpha_{index}": value
                for index, value in enumerate(alpha_by_action)
            }
        )
        metric_tensors.update(
            {
                f"metrics/beta_{index}": value
                for index, value in enumerate(beta_by_action)
            }
        )
        metric_tensors.update(
            {
                f"metrics/beta_mean_{index}": value
                for index, value in enumerate(base_dist.mean.mean(0))
            }
        )
        return metric_tensors
