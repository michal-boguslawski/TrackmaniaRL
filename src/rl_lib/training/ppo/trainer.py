"""PPO trainer orchestration, optimizer management, and rollout lifecycle."""

from logging import getLogger

import torch as T
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR
from torch.profiler import record_function

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutSettings, RolloutStep
from rl_lib.policy.masks import episode_mask
from rl_lib.run_config import RolloutSettings, RunSettings
from rl_lib.tracking.verbosity import as_verbosity, is_core_loss
from rl_lib.training.base import BuildContext
from rl_lib.training.callbacks.base import CallbackList
from rl_lib.training.ppo.losses import PPOLosses
from rl_lib.training.ppo.metrics import _tensor_metrics_to_scalars
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.update_loop import run_ppo_update_loop
from rl_lib.training.settings import (
    CosineSchedulerSettings,
    ExponentialSchedulerSettings,
    LinearSchedulerSettings,
    TrainerSettings,
)
from rl_lib.training.transitions import Transition

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
        config: TrainerSettings,
        rollout_config: RolloutSettings,
        ctx: BuildContext,
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
        self.rollout_buffer = RolloutBuffer(rollout_config, self.stack_size)
        self.rollout_config = rollout_config
        self._optimizer = self._build_optimizer(modules)
        # self._scheduler = None
        self.losses = PPOLosses(config.loss, ctx.verbosity)
        # self._callbacks = CallbackList(callbacks)

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

    def maybe_update(self, global_step: int) -> None:
        if self.rollout_buffer.is_full():
            batch = self.rollout_buffer.get()
            run_ppo_update_loop(
                batch=batch,
                epochs=self.rollout_config.epochs,
                minibatch_size=self.rollout_config.minibatch_size,
                stack_size=self.stack_size,
                cfg=self.cfg,
                training_step=global_step,
                train_step=self.train_step,
            )

            self.rollout_buffer.reset_counter()

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
        action_dist, critic_values = self.modules(observation, episode_mask(dones))
        log_probs = action_dist.log_prob(action)
        loss, metrics = self.losses.calculate_losses(
            log_probs,
            critic_values,
            action_dist,
            advantages,
            returns,
            old_log_probs,
            old_values,
        )

        self._optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.modules.parameters(), 0.5)
        self._optimizer.step()
        return {"loss/total": loss.item()}

    def _build_optimizer(self, modules: ActorCritic) -> Optimizer:
        return Adam(
            modules.parameters(),
            lr=self.cfg.optimizer.backbone_lr,
            eps=self.cfg.optimizer.eps,
        )
