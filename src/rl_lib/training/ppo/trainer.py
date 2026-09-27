"""PPO trainer orchestration, optimizer management, and rollout lifecycle."""

from logging import getLogger
from numpy.typing import NDArray
from gymnasium import Env
import torch as T
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from rl_lib.agent import Agent
from rl_lib.training.callbacks.base import Callback, CallbackList
from rl_lib.run_config import RolloutSettings, RunSettings, TrainerSettings
from rl_lib.training.ppo.losses import PPOLosses
from rl_lib.training.ppo.metrics import _tensor_metrics_to_scalars
from rl_lib.training.ppo.update_loop import run_ppo_update_loop


logger = getLogger(__name__)


class PPOTrainer:
    """PPO optimization loop with callbacks and adaptive hyperparameters.

    Manages the optimizer, scheduler, entropy coefficient decay, and the
    per-update training loop (epochs x minibatches). Callbacks receive
    metrics at start, minibatch, epoch, and end of each training call.

    Attributes:
        cfg (TrainerSettings): Training hyperparameters.
        _agent (Agent): Policy/value network wrapper.
        _optimizer: Adam or AdamW with parameter groups.
        _scheduler: LR scheduler (CosineAnnealingLR, LinearLR, or None).
        _entropy_coef: Current entropy coefficient (decays per update).
        _step: Global training step counter (minibatches processed).
    """

    def __init__(
        self,
        agent: Agent,
        config: TrainerSettings,
        callbacks: list[Callback] | None = None,
    ):
        """Initialize the PPO trainer.

        Args:
            agent: Agent wrapping the policy/value network.
            config: TrainerSettings with all PPO hyperparameters.
            callbacks: Callbacks for training events.
        """
        self.cfg = config
        self._agent = agent
        self._optimizer = self._build_optimizer()
        self._scheduler = None
        # The only trainer value that is not purely configuration: entropy_coef
        # decays every update, so its live value lives on the instance while
        # cfg.entropy_coef stays the initial coefficient.
        self._entropy_coef = self.cfg.entropy_coef
        self._losses = PPOLosses(self._agent)
        self._step = 0
        self._callbacks = CallbackList(callbacks)
        self._agent.train()

    def _build_optimizer(self) -> Adam | AdamW:
        """Build optimizer with differential LR/weight_decay per parameter group.

        Groups (from agent.network_parameter_groups):
        - backbone_decay: CNN + temporal weights (backbone_lr, weight_decay)
        - backbone_no_decay: CNN + temporal biases/norms (backbone_lr, no_decay_weight_decay)
        - head_decay: Actor/critic weights (head_lr, weight_decay)
        - head_no_decay: Actor/critic biases + log_std (head_lr, no_decay_weight_decay)

        Returns:
            Configured Adam or AdamW optimizer.
        """
        param_groups = self._agent.network_parameter_groups()
        optimizer_type = {"Adam": Adam, "AdamW": AdamW}[self.cfg.optimizer]
        return optimizer_type(
            [
                {"params": param_groups["backbone_decay"], "lr": self.cfg.backbone_lr, "weight_decay": self.cfg.weight_decay},
                {"params": param_groups["backbone_no_decay"], "lr": self.cfg.backbone_lr, "weight_decay": self.cfg.no_decay_weight_decay},
                {"params": param_groups["head_decay"], "lr": self.cfg.head_lr, "weight_decay": self.cfg.weight_decay},
                {"params": param_groups["head_no_decay"], "lr": self.cfg.head_lr, "weight_decay": self.cfg.no_decay_weight_decay},
            ],
            eps=self.cfg.optimizer_eps,
            betas=(self.cfg.optimizer_beta1, self.cfg.optimizer_beta2),
            amsgrad=self.cfg.optimizer_amsgrad,
        )

    def setup_train(self, run: RunSettings, rollout: RolloutSettings) -> None:
        """Initialize learning rate scheduler based on total training iterations.

        Called once before training starts by the entrypoint.

        Args:
            run: RunSettings with total_steps.
            rollout: RolloutSettings with buffer_size.
        """
        total_iters = run.total_steps // rollout.buffer_size + self.cfg.scheduler_extra_iters
        if self.cfg.scheduler == "cosine":
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                eta_min=self.cfg.scheduler_min_lr,
                T_max=total_iters,
            )
        elif self.cfg.scheduler == "linear":
            max_lr = max(self.cfg.backbone_lr, self.cfg.head_lr)
            self._scheduler = LinearLR(
                self._optimizer,
                start_factor=self.cfg.scheduler_start_factor,
                end_factor=self.cfg.scheduler_min_lr / max_lr,
                total_iters=total_iters,
            )
        else:
            self._scheduler = None

    @property
    def stack_size(self) -> int:
        """Temporal window size (delegated to agent)."""
        return self._agent.stack_size

    @property
    def device(self) -> T.device:
        """Compute device (delegated to agent)."""
        return self._agent.device

    def act(self, observation: T.Tensor, done: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """Sample action, log-prob, and value (delegated to agent).

        Args:
            observation: NHWC uint8 tensor (batch, H, W, C).
            done: Boolean tensor (batch,) episode termination flags.

        Returns:
            Tuple of (action, log_probs, value).
        """
        action, log_probs, critic_value = self._agent.act(observation, done)
        return action, log_probs, critic_value

    def clip_grad_norm(
        self,
        backbone_max_norm: float | None = None,
        actor_max_norm: float | None = None,
        critic_max_norm: float | None = None,
    ) -> dict[str, float]:
        """Clip gradients per parameter group and return norms.

        Args:
            backbone_max_norm: Max norm for CNN + temporal params.
            actor_max_norm: Max norm for actor params.
            critic_max_norm: Max norm for critic params.
            (Defaults from config if None)

        Returns:
            Dict with clipped norms per group.
        """
        backbone_max_norm = self.cfg.backbone_max_grad_norm if backbone_max_norm is None else backbone_max_norm
        actor_max_norm = self.cfg.actor_max_grad_norm if actor_max_norm is None else actor_max_norm
        critic_max_norm = self.cfg.critic_max_grad_norm if critic_max_norm is None else critic_max_norm
        norm_tensors = self._agent.clip_grad_norms(
            backbone_max_norm, actor_max_norm, critic_max_norm
        )
        return _tensor_metrics_to_scalars(
            {
                key: norm_tensors[key]
                for key in (
                    "grad_norm/backbone_total",
                    "grad_norm/actor_total",
                    "grad_norm/critic_total",
                )
            }
        )

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
            observation: Flattened observations (batch * seq, H, W, C).
            action: Actions taken (batch * seq, action_dim).
            old_log_probs: Log-probs under old policy (batch * seq,).
            old_values: Value estimates under old policy (batch * seq,).
            returns: GAE returns (batch * seq,).
            advantages: GAE advantages (batch * seq,).
            dones: Done flags for temporal masking (batch * seq,).

        Returns:
            Dict of loss and metric scalars for logging.
        """

        (actor_loss, critic_loss, entropy_loss), loss_metrics = self.calculate_losses(advantages, returns, old_log_probs, old_values, observation, action, dones)
        loss = actor_loss + self.cfg.critic_beta * critic_loss - self._entropy_coef * entropy_loss

        self._optimizer.zero_grad()

        if self.cfg.skip_nonfinite_updates and not T.isfinite(loss):
            logger.error(f"Non-finite loss, skipping update")
            return {}

        loss.backward()

        grad_norms = self._agent.clip_grad_norms(
            self.cfg.backbone_max_grad_norm,
            self.cfg.actor_max_grad_norm,
            self.cfg.critic_max_grad_norm,
        )
        metrics = self._get_train_step_metrics(loss, grad_norms)

        self._optimizer.step()
        self._step += 1

        return {**loss_metrics, **metrics}

    def _get_train_step_metrics(
        self, loss: T.Tensor, grad_norms: dict[str, T.Tensor]
    ) -> dict[str, float]:
        """Collect loss and gradient diagnostics with one scalar transfer."""
        metric_tensors = {
            "loss/total": loss.detach(),
            **{
                key: grad_norms[key]
                for key in (
                    "grad_norm/cnn",
                    "grad_norm/sequence_encoder",
                    "grad_norm/actor",
                    "grad_norm/critic",
                    "grad_norm/max",
                )
            },
        }
        return _tensor_metrics_to_scalars(metric_tensors)

    def calculate_losses(
        self,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None
    ) -> tuple[tuple[T.Tensor, T.Tensor, T.Tensor], dict[str, float]]:
        """Delegate PPO objective calculation to the loss component.

        Args:
            advantages: GAE advantages (batch * seq,).
            returns: GAE returns (batch * seq,).
            old_log_probs: Old policy log-probs (batch * seq,).
            old_values: Old value estimates (batch * seq,).
            observation: Flattened observations (batch * seq, H, W, C).
            action: Actions taken (batch * seq, action_dim).
            dones: Done flags for temporal masking (batch * seq,) or None.

        Returns:
            Tuple of ((actor_loss, critic_loss, entropy_loss), metrics_dict).
        """
        return self._losses.calculate_losses(
            self.cfg,
            advantages,
            returns,
            old_log_probs,
            old_values,
            observation,
            action,
            dones,
        )

    def _on_start(self, *args, **kwargs):
        self._callbacks.on_start(*args, **kwargs)

    def _on_minibatch(self, *args, **kwargs):
        self._callbacks.on_minibatch(*args, **kwargs)

    def _on_end(self, *args, **kwargs):
        self._callbacks.on_end(*args, **kwargs)

    def _on_epoch(self, *args, **kwargs):
        self._callbacks.on_epoch(*args, **kwargs)

    def _get_metrics_from_batch(self, batch: T.Tensor):
        """Compute rollout-level metrics from a full batch for logging."""
        metrics = {
            "rollout/returns": batch["returns"].mean().item(),
            "rollout/advantages_mean": batch["advantages"].mean().item(),
            "rollout/advantages_std": batch["advantages"].std().item(),
            "rollout/critic_values": batch["critic_value"].mean().item(),
            "rollout/old_log_probs" : batch["old_log_probs"].sum(-1).mean().item(),
        }
        action_means = batch["action"].mean((0, 1))
        for i, mean in enumerate(action_means):
            metrics[f"rollout/action_mean_{i}"] = mean.item()
        action_stds = batch["action"].std((0, 1), unbiased=True)
        for i, std in enumerate(action_stds):
            metrics[f"rollout/action_std_{i}"] = std.item()

        explained_variance = 1 - (batch["returns"] - batch["critic_value"]).var() / (batch["returns"].var() + self.cfg.explained_variance_epsilon)
        metrics["rollout/explained_variance"] = explained_variance
        return metrics

    def step_env(self, env: Env, state: NDArray, done: T.Tensor, temperature: float | None = None) -> tuple:
        """Step environment using agent policy (delegated to agent)."""
        return self._agent.step_env(env, state, done, temperature)

    def bootstrap_value(self, observation: T.Tensor) -> T.Tensor:
        """Bootstrap value for truncated episodes (delegated to agent)."""
        return self._agent.bootstrap_value(observation)

    def train(
        self,
        batch: dict[str, T.Tensor],
        epochs: int,
        minibatch_size: int,
        training_step: int,
    ) -> None:
        """Run PPO epochs over a collected batch.

        Args:
            batch: Dict from RolloutBuffer.get().
            epochs: Number of passes over the batch.
            minibatch_size: Transitions per minibatch.
            training_step: Global environment step count (for callbacks/logging).
        """

        self._on_start(
            step=training_step,
            metrics=self._get_metrics_from_batch(batch)
        )

        if self.cfg.advantage_normalization_strategy == "global":
            batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (
                batch["advantages"].std() + self.cfg.advantage_epsilon
            )

        run_ppo_update_loop(
            batch=batch,
            epochs=epochs,
            minibatch_size=minibatch_size,
            stack_size=self.stack_size,
            cfg=self.cfg,
            training_step=training_step,
            train_step=self.train_step,
            current_update_step=lambda: self._step,
            on_minibatch=self._on_minibatch,
            on_epoch=self._on_epoch,
        )

        if self._scheduler:
            self._scheduler.step()
        self._entropy_coef = max(self.cfg.entropy_decay * self._entropy_coef, self.cfg.entropy_coef_min)
        metrics = {
            "training/entropy_coef": self._entropy_coef,
            # "training/lr": self._optimizer.param_groups[0]["lr"]
        }
        for i, pg in enumerate(self._optimizer.param_groups):
            metrics[f"training/lr_{i}"] = pg["lr"]
        self._on_end(metrics=metrics, step=training_step)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ppo_epsilon={self.cfg.ppo_epsilon}, "
            f"critic_beta={self.cfg.critic_beta}, "
            f"entropy_coef={self._entropy_coef}, "
            f"entropy_decay={self.cfg.entropy_decay}, "
            f"target_kl={self.cfg.target_kl}, "
            f"kl_warmup_steps={self.cfg.kl_warmup_steps}, "
            f"backbone_lr={self.cfg.backbone_lr}, "
            f"head_lr={self.cfg.head_lr}, "
            f"weight_decay={self.cfg.weight_decay}, "
            f"hard_stop_kl={self.cfg.hard_stop_kl}, "
            f"advantage_normalization_strategy={self.cfg.advantage_normalization_strategy!r}, "
            f"scheduler={self._scheduler.__class__.__name__ if self._scheduler else None}, "
            f"mean_reg_coef={self.cfg.mean_reg_coef})"
        )

    def config(self) -> dict[str, int | float | str]:
        """Hyperparameters, suitable for mlflow.log_params (with a prefix)."""
        config = {
            **{
                f"{key}": ("none" if value is None else value)
                for key, value in self.cfg.model_dump().items()
                if isinstance(value, (int, float, str, bool)) or value is None
            },
            "entropy_coef_init": self.cfg.entropy_coef,
            "entropy_coef_current": self._entropy_coef,
            "optimizer": self._optimizer.__class__.__name__,
            "scheduler_config": self.cfg.scheduler,
            "scheduler": self._scheduler.__class__.__name__ if self._scheduler else "none",
        }
        return config

    def network_config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable network hyperparameters."""
        return self._agent.network_config()
