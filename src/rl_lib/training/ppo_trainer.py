"""PPO Trainer: computes losses, performs optimization steps, manages callbacks.

Implements the PPO clipped objective with:
- Clipped surrogate loss for actor
- Clipped value loss (Huber) for critic
- Entropy bonus with decay
- Mean action regularization (penalizes distance from neutral action)
- KL-based early stopping (hard and soft)
- Advantage normalization (batch or global)
- Learning rate scheduling (cosine or linear)
- Differential learning rates/weight decay per parameter group
"""

import numpy as np
from numpy.typing import NDArray
from logging import getLogger
from gymnasium import Env
import torch as T
from torch import nn
from torch.optim import Adam, AdamW
from torch.distributions import Distribution
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from typing import Iterator

from rl_lib.agent import Agent
from rl_lib.training.callbacks.base import Callback, CallbackList
from rl_lib.run_config import RolloutSettings, RunSettings, TrainerSettings


logger = getLogger(__name__)


def _tensor_metrics_to_scalars(metrics: dict[str, T.Tensor]) -> dict[str, float]:
    """Move scalar metric tensors to Python in one device synchronization."""
    if not metrics:
        return {}
    keys = tuple(metrics)
    values = T.stack([metrics[key].detach().reshape(()) for key in keys])
    scalars = values.cpu().tolist()
    return dict(zip(keys, scalars, strict=True))


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
        self._critic_loss_fn = nn.HuberLoss(
            reduction="none", delta=self._agent.network.cfg.critic.huber_delta
        )
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

    def _actor_loss(self, advantages: T.Tensor, log_probs: T.Tensor, old_log_probs: T.Tensor, action_mean: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        """Compute clipped surrogate actor loss with mean regularization.

        Args:
            advantages: GAE advantages (batch * seq,).
            log_probs: New policy log-probs (batch * seq,).
            old_log_probs: Old policy log-probs (batch * seq,).
            action_mean: Mean actions from new policy (batch * seq, action_dim).

        Returns:
            Tuple of (actor_loss, metrics_dict).
        """
        if self.cfg.advantage_normalization_strategy and self.cfg.advantage_normalization_strategy == "batch":
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.cfg.advantage_epsilon)

        assert log_probs.shape == old_log_probs.shape
        log_ratio = (log_probs - old_log_probs).clamp(self.cfg.log_ratio_min, self.cfg.log_ratio_max)

        ratio = log_ratio.sum(-1).clamp(-5, 5).exp()

        clipped_ratio = ratio.clamp(
            min=1 - self.cfg.ppo_epsilon,
            max=1 + self.cfg.ppo_epsilon
        )

        assert advantages.shape == ratio.shape
        surrogate_loss = -T.minimum(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Penalize distance from the no-input action [steer=0, gas=0, brake=0].
        # The midpoint of the gas/brake ranges is not neutral: it applies both.
        mean_reg = action_mean.pow(2).mean()
        actor_loss = surrogate_loss + self.cfg.mean_reg_coef * mean_reg

        metrics = self._actor_loss_metrics(
            log_ratio, surrogate_loss, mean_reg, action_mean, actor_loss
        )

        return actor_loss, metrics

    def _actor_loss_metrics(
        self,
        log_ratio: T.Tensor,
        surrogate_loss: T.Tensor,
        mean_reg: T.Tensor,
        action_mean: T.Tensor,
        actor_loss: T.Tensor,
    ) -> dict[str, float]:
        """Compute detailed actor metrics for logging."""
        with T.no_grad():
            log_ratio_total = log_ratio.sum(-1)
            ratio_total = log_ratio_total.exp()
            approx_kl = (ratio_total - 1 - log_ratio_total).mean()
            ratio_per_action = log_ratio.exp()
            approx_kl_per_action = (ratio_per_action - 1 - log_ratio).mean(dim=0)
            ratio_max = ratio_total.max()
            clip_fraction = (T.abs(ratio_total - 1) > self.cfg.ppo_epsilon).float().mean()

        with T.no_grad():
            metric_tensors = {
                "metrics/actor_loss": actor_loss.detach(),
                "metrics/approx_kl": approx_kl,
                "metrics/ratio_max": ratio_max,
                "metrics/clip_fraction": clip_fraction,
                "metrics/surrogate_loss": surrogate_loss.detach(),
                "metrics/mean_reg": mean_reg.detach(),
                "metrics/mean_abs_max": action_mean.abs().max(),
                "metrics/tanh_saturation_frac": (
                    action_mean.abs() > self.cfg.tanh_saturation_threshold
                ).float().mean(),
            }
            metric_tensors.update(
                {
                    f"metrics/mean_abs_max_{i}": action_mean[:, i].abs().max()
                    for i in range(action_mean.shape[-1])
                }
            )
            metric_tensors.update(
                {
                    f"metrics/approx_kl_{i}": value
                    for i, value in enumerate(approx_kl_per_action)
                }
            )

        return _tensor_metrics_to_scalars(metric_tensors)

    def _critic_loss(self, returns: T.Tensor, values: T.Tensor, old_values: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        """Compute clipped Huber value loss.

        Args:
            returns: GAE returns (batch * seq,).
            values: New value estimates (batch * seq,).
            old_values: Old value estimates (batch * seq,).

        Returns:
            Tuple of (critic_loss, metrics_dict).
        """
        assert values.shape == old_values.shape == returns.shape
        value_clip_epsilon = self.cfg.value_clip_epsilon or self.cfg.ppo_epsilon
        clipped_values = old_values + (values - old_values).clamp(-value_clip_epsilon, value_clip_epsilon)
        loss_unclipped = self._critic_loss_fn(values, returns)
        loss_clipped = self._critic_loss_fn(clipped_values, returns)
        
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()

        return critic_loss, _tensor_metrics_to_scalars(
            {"loss/critic": critic_loss.detach()}
        )

    def _entropy_loss(self, dist: Distribution) -> tuple[T.Tensor, dict[str, float]]:
        """Compute entropy bonus (negative loss = maximize entropy).

        Uses base Beta distribution entropy (TanhTransform has no closed form).

        Args:
            dist: Actor's TransformedDistribution.

        Returns:
            Tuple of (entropy_loss, metrics_dict). entropy_loss is positive,
            subtracted in total loss to encourage exploration.
        """
        # entropy of the base Normal; TanhTransform doesn't have closed-form entropy
        entropy: T.Tensor = dist.base_dist.entropy()
        entropy_loss = entropy.sum(dim=-1).mean()

        with T.no_grad():
            log_std = dist.base_dist.variance.pow(1/2).log().mean(0)
            metric_tensors = {
                **{
                    f"metrics/entropy_{i}": value
                    for i, value in enumerate(entropy.mean(0))
                },
                "loss/entropy": entropy_loss.detach(),
                **{
                    f"metrics/log_std_{i}": value
                    for i, value in enumerate(log_std)
                },
            }
        metrics = _tensor_metrics_to_scalars(metric_tensors)

        return entropy_loss, metrics


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
        """Evaluate policy on batch and compute all loss components.

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
        log_probs, values, dist, action_mean = self._agent.evaluate_actions(observation, action, dones)
        actor_loss, actor_metrics = self._actor_loss(advantages, log_probs, old_log_probs, action_mean)
        critic_loss, critic_metrics = self._critic_loss(returns, values, old_values)
        entropy_loss, entropy_metrics = self._entropy_loss(dist)
        metrics = {**actor_metrics, **critic_metrics, **entropy_metrics}
        return (actor_loss, critic_loss, entropy_loss), metrics

    def _on_start(self, *args, **kwargs):
        self._callbacks.on_start(*args, **kwargs)

    def _on_minibatch(self, *args, **kwargs):
        self._callbacks.on_minibatch(*args, **kwargs)

    def _on_end(self, *args, **kwargs):
        self._callbacks.on_end(*args, **kwargs)

    def _on_epoch(self, *args, **kwargs):
        self._callbacks.on_epoch(*args, **kwargs)

    @staticmethod
    def _get_iid_minibatches(
        batch: dict[str, T.Tensor],
        minibatch_size: int,
        stack_size: int,
        shuffle: bool,
    ) -> Iterator[dict[str, T.Tensor]]:
        """Iterate over minibatches from a batch of sequences.

        Samples IID minibatches across (env, time) by flattening the batch
        and randomly permuting. For each minibatch, reconstructs the
        temporal observation/done windows of length stack_size.

        Args:
            batch: Dict from RolloutBuffer.get() with keys:
                observation: (num_envs, seq_len + stack_size - 2, H, W, C)
                action: (num_envs, seq_len - 1, action_dim)
                old_log_probs: (num_envs, seq_len - 1)
                critic_value: (num_envs, seq_len - 1)
                returns: (num_envs, seq_len - 1)
                advantages: (num_envs, seq_len - 1)
                dones: (num_envs, seq_len + stack_size - 2)
            minibatch_size: Number of transitions per minibatch.
            stack_size: Temporal window size.
            shuffle: Whether to shuffle indices (IID) or use sequential.

        Yields:
            Minibatch dict with same keys but flattened batch dimension
            and observation/dones windows of length stack_size.
        """
        num_envs, batch_size, _ = batch["action"].shape
        device = batch["action"].device
        flat_order = (
            np.random.permutation(batch_size * num_envs)
            if shuffle
            else np.arange(batch_size * num_envs)
        )
        indices = T.as_tensor(
            flat_order,
            dtype=T.long,
            device=device,
        )
        window_offsets = T.arange(stack_size, device=device)
        for index in range(0, batch_size * num_envs, minibatch_size):
            flat_indices = indices[index : index + minibatch_size]
            env_indices = flat_indices.remainder(num_envs)
            time_indices = flat_indices.div(num_envs, rounding_mode="floor")
            window_indices = time_indices[:, None] + window_offsets

            observation_windows = batch["observation"][
                env_indices[:, None], window_indices
            ]
            done_windows = batch["dones"][env_indices[:, None], window_indices]

            yield {
                "observation": observation_windows.reshape(
                    -1, *batch["observation"].shape[2:]
                ),
                "action": batch["action"][env_indices, time_indices],
                "old_log_probs": batch["old_log_probs"][env_indices, time_indices],
                "old_values": batch["critic_value"][env_indices, time_indices],
                "returns": batch["returns"][env_indices, time_indices],
                "advantages": batch["advantages"][env_indices, time_indices],
                "dones": done_windows.reshape(-1),
            }

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
        # rng: np.random.Generator,
    ):
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

        if self.cfg.advantage_normalization_strategy and self.cfg.advantage_normalization_strategy == "global":
            batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + self.cfg.advantage_epsilon)

        for epoch in range(epochs):
            epoch_kls = []
            for minibatch in self._get_iid_minibatches(
                batch,
                minibatch_size,
                self.stack_size,
                shuffle=self.cfg.minibatch_indexing_mode == "iid",
            ):
                try:
                    minibatch_metrics = self.train_step(**minibatch)
                except Exception as e:
                    logger.error(f"Error at training step {training_step}, epoch {epoch}: {e}")
                    raise e
                if not minibatch_metrics:
                    continue
                epoch_kls.append(minibatch_metrics["metrics/approx_kl"])

                if (
                    minibatch_metrics["metrics/approx_kl"] > self.cfg.target_kl * self.cfg.kl_stop_multiplier and
                    training_step > self.cfg.kl_warmup_steps and
                    self.cfg.hard_stop_kl
                ):
                    logger.warning(f"Hard stop mid-epoch: KL {minibatch_metrics['metrics/approx_kl']:.4f}")
                    break

                self._on_minibatch(metrics=minibatch_metrics, step=self._step)
            mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else None
            self._on_epoch()
            if (
                mean_epoch_kl is not None
                and mean_epoch_kl > self.cfg.target_kl
                and training_step > self.cfg.kl_warmup_steps
                and self.cfg.hard_stop_kl
            ):
                logger.warning(f"Early stop epoch {epoch}: KL {mean_epoch_kl:.4f} > {self.cfg.target_kl}")
                break

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
