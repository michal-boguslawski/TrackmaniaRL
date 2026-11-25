"""PPO trainer orchestration, optimizer management, and rollout lifecycle."""

from logging import getLogger

import torch as T
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.policy.masks import episode_mask
from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.ppo.losses import PPOLosses
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.update_loop import run_ppo_update_loop
from rl_lib.training.schedulers.factory import build_scheduler
from rl_lib.training.transitions import Transition
from rl_lib.training.utils import build_optimizer

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

            self.rollout_buffer.reset_counter()
            if self._callbacks:
                metrics = {
                    f"optimization/lr_{pg['name']}": pg["lr"]
                    for pg in self._optimizer.param_groups
                }
                self._callbacks.on_event(
                    CallbackEvent.TRAIN_END, step=global_step, metrics=metrics
                )

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
        loss, metrics = self.losses.calculate_losses(
            log_probs,
            critic_values,
            action_dist,
            advantages,
            returns,
            old_log_probs,
            old_values,
            action_mean,
        )

        self._optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.modules.parameters(), 0.5)
        self._optimizer.step()
        return {"loss/total": loss.item(), **metrics}
