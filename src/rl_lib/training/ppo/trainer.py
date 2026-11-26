"""PPO trainer orchestration, optimizer management, and rollout lifecycle."""

from dataclasses import fields
from logging import getLogger

import torch as T
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.policy.masks import episode_mask
from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.training.metrics.contexts import OptimizationContext, RolloutContext
from rl_lib.training.metrics.factory import MetricsFactory
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.ppo.losses import PPOLosses
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.update_loop import run_ppo_update_loop
from rl_lib.training.schedulers.factory import build_scheduler
from rl_lib.training.transitions import Transition
from rl_lib.training.utils import build_optimizer, tensor_metrics_to_scalars

logger = getLogger(__name__)


class PPOTrainer:
    """PPO optimization loop with callbacks and adaptive hyperparameters.

    Manages the optimizer, scheduler, entropy coefficient decay, and the
    per-update training loop (epochs x minibatches). Callbacks receive
    metrics at start, minibatch, epoch, and end of each training call.

    Attributes:
        # TODO
    """

    def __init__(
        self,
        modules: ActorCritic,
        config: PPOTrainerSettings,
        total_steps: int,
        metrics_factory: MetricsFactory | None = None,
        callbacks: Callback | None = None,
    ):
        """Initialize the PPO trainer.

        Args:
            agent: Agent wrapping the policy/value network.
            config: TrainerSettings with all PPO hyperparameters.
            callbacks: Callbacks for training events.
            verbosity: Metric detail level deciding which metrics are computed
                and handed to the tracking backends.

        Raises:
            ValueError: If verbosity is not one of 0, 1, or 2.
        """
        self.cfg = config
        self.modules = modules
        self.stack_size = self.modules.encoder.stack_size
        self.rollout_config = config.rollout
        self.rollout_buffer = RolloutBuffer(self.rollout_config, self.stack_size)
        self._optimizer: Optimizer = build_optimizer(
            self.modules.optimizer_parameter_groups(
                self.cfg.optimizer.learning_rates,
                self.cfg.optimizer.weight_decay,
                self.cfg.optimizer.no_decay_weight_decay,
            ),
            self.cfg.optimizer.name,
            self.cfg.optimizer.eps,
        )
        self._callbacks = callbacks
        self._scheduler = build_scheduler(
            self._optimizer,
            self.cfg.scheduler,
            total_steps=total_steps,
            steps_per_training=self.cfg.rollout.steps_per_env,
        )
        self.losses = PPOLosses(config.loss)
        self.metrics_factory = metrics_factory

    def set_callbacks(self, callbacks: Callback) -> None:
        self._callbacks = callbacks

    def observe(self, t: Transition) -> None:
        with T.no_grad():
            value = self.modules.critic(t.extras["temporal"])
        self.rollout_buffer.add(
            RolloutStep(
                observation=t.obs,
                action=t.action,
                old_log_probs=t.extras["log_probs"],
                reward=t.reward,
                terminated=t.terminated,
                truncated=t.truncated,
                critic_value=value,
            )
        )

    def maybe_update(self, global_step: int) -> bool:
        if self.rollout_buffer.is_full():
            if self._callbacks:
                self._callbacks.on_event(CallbackEvent.TRAIN_START, step=global_step)
            batch = self.rollout_buffer.get()

            rollout_context = RolloutContext(
                batch["action"],
                batch["returns"],
                batch["advantages"],
                batch["dones"],
                batch["critic_value"],
            )
            rollout_metrics = self.metrics_factory.rollout.calculate(rollout_context)
            self._callbacks.on_event(
                CallbackEvent.TRAIN_START,
                step=global_step,
                metrics=rollout_metrics,
            )
            self.modules.train()
            run_ppo_update_loop(
                batch=batch,
                stack_size=self.stack_size,
                optimization_settings=self.cfg.optimization,
                training_step=global_step,
                train_step=self.train_step,
                metrics_logger=self._callbacks,
                kl_control_settings=self.cfg.kl,
            )

            self._callbacks.on_event(CallbackEvent.TRAIN_END, step=global_step)
            self.rollout_buffer.reset_counter()

            if self._scheduler:
                self._scheduler.step()

            return True
        return False

    def train_step(
        self,
        observation: T.Tensor,
        action: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        returns: T.Tensor,
        advantages: T.Tensor,
        dones: T.Tensor,
    ) -> dict[str, float]:
        """Single minibatch gradient step.

        Args:
            observation: Flattened observations (batch, seq, H, W, C).
            action: Actions taken (batch, action_dim).
            old_log_probs: Log-probs under old policy (batch, action_dim).
            old_values: Value estimates under old policy (batch,).
            returns: GAE returns (batch,).
            advantages: GAE advantages (batch,).
            dones: Done flags for temporal masking (batch, seq).

        Returns:
            Dict of loss and metric scalars for logging.
        """
        action_dist, critic_values, action_mean = self.modules(
            observation, episode_mask(dones)
        )
        log_probs = action_dist.log_prob(action)

        update_context = PPOUpdateContext(
            values=critic_values,
            returns=returns,
            dist=action_dist,
            old_log_probs=old_log_probs,
            new_log_probs=log_probs,
            advantages=advantages,
            actions=action_mean,
            old_values=old_values,
            actor_loss_cfg=self.cfg.loss.actor_loss_cfg,
            entropy_decay=self.losses.entropy_decay,
        )
        loss = self.losses.calculate_losses(update_context)
        with T.no_grad():
            update_metrics = self.metrics_factory.update.calculate(update_context)

        self._optimizer.zero_grad()
        loss.total.backward()

        optimization_context = OptimizationContext(optimizer=self._optimizer)
        clip_grad_norm_(self.modules.parameters(), 0.5)
        with T.no_grad():
            optimization_metrics = self.metrics_factory.optimization.calculate(
                optimization_context
            )
        self._optimizer.step()
        self.losses.entropy_decay.step()

        return {
            **tensor_metrics_to_scalars(
                {f.name: getattr(loss, f.name) for f in fields(loss)}
            ),
            **update_metrics,
            **optimization_metrics,
        }
