"""PPO trainer orchestration, optimizer management, and rollout lifecycle."""

from logging import getLogger
from numpy.typing import NDArray
from gymnasium import Env
import torch as T
from torch.profiler import record_function
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR

from rl_lib.agent import Agent
from rl_lib.tracking.verbosity import Verbosity, as_verbosity, is_core_loss
from rl_lib.training.callbacks.base import Callback, CallbackList
from rl_lib.run_config import RolloutSettings, RunSettings
from rl_lib.training.settings import (
    CosineSchedulerSettings,
    ExponentialSchedulerSettings,
    LinearSchedulerSettings,
    TrainerSettings,
)
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
        _scheduler: Cosine, linear, exponential, or disabled learning-rate scheduler.
        _entropy_coef: Current entropy coefficient (decays per update).
        _step: Global training step counter (minibatches processed).
        _verbosity (Verbosity): Metric detail; below ALL the diagnostic
            metrics are not computed at all.
        last_step_had_truncation (bool): Whether the latest environment step
            truncated at least one environment.
    """

    def __init__(
        self,
        agent: Agent,
        config: TrainerSettings,
        callbacks: list[Callback] | None = None,
        verbosity: int | Verbosity = Verbosity.ALL,
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
        self._agent = agent
        self._optimizer = self._build_optimizer()
        self._scheduler = None
        # Entropy decays every update, so its live value stays on the instance.
        self._entropy_coef = self.cfg.loss.entropy.coefficient
        self._verbosity = as_verbosity(verbosity)
        self._losses = PPOLosses(self._agent, self._verbosity)
        self._step = 0
        self._callbacks = CallbackList(callbacks)
        self._defer_diagnostics = False
        self._pending_diagnostics: list[tuple[dict[str, T.Tensor], int]] = []
        self._agent.train()

    def _build_optimizer(self) -> Adam | AdamW:
        """Build optimizer with differential LR/weight_decay per parameter group.

        Groups (from agent.network_parameter_groups):
        - cnn_decay/no_decay: CNN weights and biases/norms
        - sequence_encoder_decay/no_decay: sequence weights and biases/norms
        - head_decay: Actor/critic weights (head_lr, weight_decay)
        - head_no_decay: Actor/critic biases + log_std (head_lr, no_decay_weight_decay)

        CNN and sequence encoder rates independently override backbone_lr when
        their corresponding settings are configured.

        Returns:
            Configured Adam or AdamW optimizer. Uses the fused CUDA kernel when
            trainer.optimizer.fused is set and the agent runs on CUDA, which
            replaces the per-parameter foreach kernel sequence with one launch.
        """
        param_groups = self._agent.network_parameter_groups()
        optimizer_cfg = self.cfg.optimizer
        optimizer_type = {"Adam": Adam, "AdamW": AdamW}[optimizer_cfg.name]
        cnn_lr = optimizer_cfg.cnn_lr or optimizer_cfg.backbone_lr
        sequence_encoder_lr = optimizer_cfg.sequence_encoder_lr or optimizer_cfg.backbone_lr
        implementation_kwargs = {}
        if optimizer_cfg.fused and self._agent.device.type == "cuda":
            implementation_kwargs["fused"] = True
        return optimizer_type(
            [
                {
                    "params": param_groups["cnn_decay"],
                    "lr": cnn_lr,
                    "weight_decay": optimizer_cfg.weight_decay,
                },
                {
                    "params": param_groups["cnn_no_decay"],
                    "lr": cnn_lr,
                    "weight_decay": optimizer_cfg.no_decay_weight_decay,
                },
                {
                    "params": param_groups["sequence_encoder_decay"],
                    "lr": sequence_encoder_lr,
                    "weight_decay": optimizer_cfg.weight_decay,
                },
                {
                    "params": param_groups["sequence_encoder_no_decay"],
                    "lr": sequence_encoder_lr,
                    "weight_decay": optimizer_cfg.no_decay_weight_decay,
                },
                {
                    "params": param_groups["head_decay"],
                    "lr": optimizer_cfg.head_lr,
                    "weight_decay": optimizer_cfg.weight_decay,
                },
                {
                    "params": param_groups["head_no_decay"],
                    "lr": optimizer_cfg.head_lr,
                    "weight_decay": optimizer_cfg.no_decay_weight_decay,
                },
            ],
            eps=optimizer_cfg.eps,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            amsgrad=optimizer_cfg.amsgrad,
            **implementation_kwargs,
        )

    def setup_train(self, run: RunSettings, rollout: RolloutSettings) -> None:
        """Initialize learning rate scheduler based on total training iterations.

        Called once before training starts by the entrypoint.

        Args:
            run: RunSettings with total_steps.
            rollout: RolloutSettings with buffer_size.
        """
        scheduler_cfg = self.cfg.scheduler
        if isinstance(scheduler_cfg, CosineSchedulerSettings):
            total_iters = run.total_steps // rollout.buffer_size + scheduler_cfg.extra_iters
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                eta_min=scheduler_cfg.eta_min,
                T_max=total_iters,
            )
        elif isinstance(scheduler_cfg, LinearSchedulerSettings):
            total_iters = run.total_steps // rollout.buffer_size + scheduler_cfg.extra_iters
            optimizer_cfg = self.cfg.optimizer
            max_lr = max(
                optimizer_cfg.cnn_lr or optimizer_cfg.backbone_lr,
                optimizer_cfg.sequence_encoder_lr or optimizer_cfg.backbone_lr,
                optimizer_cfg.head_lr,
            )
            self._scheduler = LinearLR(
                self._optimizer,
                start_factor=scheduler_cfg.start_factor,
                end_factor=scheduler_cfg.min_lr / max_lr,
                total_iters=total_iters,
            )
        elif isinstance(scheduler_cfg, ExponentialSchedulerSettings):
            self._scheduler = ExponentialLR(self._optimizer, gamma=scheduler_cfg.gamma)
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

    @property
    def last_step_had_truncation(self) -> bool:
        """Whether the most recent environment step truncated any environment."""
        return self._agent.last_step_had_truncation

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
        backbone_max_norm = self.cfg.gradients.backbone_max_grad_norm if backbone_max_norm is None else backbone_max_norm
        actor_max_norm = self.cfg.gradients.actor_max_grad_norm if actor_max_norm is None else actor_max_norm
        critic_max_norm = self.cfg.gradients.critic_max_grad_norm if critic_max_norm is None else critic_max_norm
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
        loss, loss_metric_tensors = self._calculate_total_loss(
            advantages,
            returns,
            old_log_probs,
            old_values,
            observation,
            action,
            dones,
        )
        self._optimizer.zero_grad()
        essential_metrics = self._essential_metrics(loss, loss_metric_tensors)
        if essential_metrics is None:
            return {}

        grad_norms = self._backward_and_clip(loss)
        diagnostics = self._gradient_diagnostics(
            loss, loss_metric_tensors, essential_metrics, grad_norms
        )
        cnn_parameters_before = self._snapshot_cnn_parameters()
        self._optimizer_step()
        diagnostics.update(self._cnn_update_metrics(cnn_parameters_before))
        self._emit_diagnostics(diagnostics, essential_metrics, action.shape[0])
        self._step += 1

        return essential_metrics

    def _calculate_total_loss(
        self,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Evaluate the PPO losses and combine them into the optimized scalar."""
        with record_function("ppo/forward_and_loss"):
            (actor_loss, critic_loss, entropy_loss), metric_tensors = (
                self.calculate_losses(
                    advantages,
                    returns,
                    old_log_probs,
                    old_values,
                    observation,
                    action,
                    dones,
                    return_tensors=True,
                )
            )
        total_loss = (
            actor_loss
            + self.cfg.loss.critic_beta * critic_loss
            - self._entropy_coef * entropy_loss
        )
        return total_loss, metric_tensors

    def _essential_metrics(
        self, loss: T.Tensor, metric_tensors: dict[str, T.Tensor]
    ) -> dict[str, float] | None:
        """Synchronize control-flow and core-loss metrics, or reject bad loss."""
        essential_tensors = {
            key: value
            for key, value in metric_tensors.items()
            if key == "metrics/approx_kl"
            or (self._verbosity.logs_core_losses and is_core_loss(key))
        }
        if self.cfg.stability.skip_nonfinite_updates:
            essential_tensors["_control/loss_is_finite"] = T.isfinite(loss).to(
                dtype=T.float32
            )
        essential_metrics = _tensor_metrics_to_scalars(essential_tensors)
        if self.cfg.stability.skip_nonfinite_updates and not essential_metrics.pop(
            "_control/loss_is_finite"
        ):
            logger.error("Non-finite loss, skipping update")
            return None
        return essential_metrics

    def _backward_and_clip(self, loss: T.Tensor) -> dict[str, T.Tensor]:
        """Backpropagate and clip gradients by configured parameter group."""
        with record_function("ppo/backward"):
            loss.backward()
        return self._agent.clip_grad_norms(
            self.cfg.gradients.backbone_max_grad_norm,
            self.cfg.gradients.actor_max_grad_norm,
            self.cfg.gradients.critic_max_grad_norm,
        )

    def _gradient_diagnostics(
        self,
        loss: T.Tensor,
        metric_tensors: dict[str, T.Tensor],
        essential_metrics: dict[str, float],
        grad_norms: dict[str, T.Tensor],
    ) -> dict[str, T.Tensor]:
        """Collect optional loss, module-gradient, and clipping diagnostics."""
        if not self._verbosity.logs_diagnostics:
            return {}
        diagnostics = {
            key: value
            for key, value in metric_tensors.items()
            if key not in essential_metrics
        }
        diagnostics["loss/total"] = loss.detach()
        diagnostics.update(
            {
                key: grad_norms[key]
                for key in (
                    "grad_norm/cnn",
                    "grad_norm/sequence_encoder",
                    "grad_norm/actor",
                    "grad_norm/critic",
                    "grad_norm/max",
                    "grad_norm/backbone_total",
                    "grad_clip/backbone_scale",
                    "grad_clip/backbone_clipped",
                )
            }
        )
        diagnostics.update(
            {
                key: value
                for key, value in grad_norms.items()
                if key.startswith("grad_norm/cnn_")
            }
        )
        return diagnostics

    def _snapshot_cnn_parameters(self) -> list[T.Tensor] | None:
        """Snapshot CNN weights only when actual-update diagnostics are enabled."""
        if not self._verbosity.logs_diagnostics:
            return None
        # This measures AdamW's update, including adaptive scaling and weight decay.
        return [
            parameter.detach().clone()
            for parameter in self._agent.network.cnn.parameters()
        ]

    def _optimizer_step(self) -> None:
        """Apply the optimizer update."""
        with record_function("ppo/optimizer_step"):
            self._optimizer.step()

    def _cnn_update_metrics(
        self, parameters_before: list[T.Tensor] | None
    ) -> dict[str, T.Tensor]:
        """Measure the CNN parameter update relative to its pre-step weights."""
        if parameters_before is None:
            return {}
        parameters_after = list(self._agent.network.cnn.parameters())
        update_norms = T._foreach_norm(
            [
                parameter.detach() - previous
                for parameter, previous in zip(
                    parameters_after, parameters_before, strict=True
                )
            ],
            2,
        )
        weight_norms = T._foreach_norm(parameters_before, 2)
        update_norm = T.linalg.vector_norm(T.stack(update_norms), ord=2)
        weight_norm = T.linalg.vector_norm(T.stack(weight_norms), ord=2)
        return {
            "update_norm/cnn": update_norm,
            "update_relative/cnn": update_norm / (weight_norm + 1e-8),
        }

    def _emit_diagnostics(
        self,
        diagnostics: dict[str, T.Tensor],
        essential_metrics: dict[str, float],
        batch_size: int,
    ) -> None:
        """Defer diagnostics to the epoch boundary or convert and emit now."""
        if not diagnostics:
            return
        if self._defer_diagnostics:
            self._pending_diagnostics.append((diagnostics, batch_size))
        else:
            essential_metrics.update(_tensor_metrics_to_scalars(diagnostics))

    def calculate_losses(
        self,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None,
        return_tensors: bool = False,
    ) -> tuple[
        tuple[T.Tensor, T.Tensor, T.Tensor], dict[str, float] | dict[str, T.Tensor]
    ]:
        """Delegate PPO objective calculation to the loss component.

        Args:
            advantages: GAE advantages (batch * seq,).
            returns: GAE returns (batch * seq,).
            old_log_probs: Old policy log-probs (batch * seq,).
            old_values: Old value estimates (batch * seq,).
            observation: Flattened observations (batch * seq, H, W, C).
            action: Actions taken (batch * seq, action_dim).
            dones: Done flags for temporal masking (batch * seq,) or None.
            return_tensors: Keep scalar metrics on-device for the optimizer path.

        Returns:
            Tuple of ((actor_loss, critic_loss, entropy_loss), metrics_dict).
        """
        if return_tensors:
            return self._losses._calculate_losses(
                self.cfg,
                advantages,
                returns,
                old_log_probs,
                old_values,
                observation,
                action,
                dones,
            )
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
        metrics: dict[str, float] = {}
        if self._pending_diagnostics:
            entries, weights = zip(*self._pending_diagnostics, strict=True)
            weight_tensors = [
                T.as_tensor(weight, device=next(iter(entry.values())).device)
                for entry, weight in zip(entries, weights, strict=True)
            ]
            total_weight = T.stack(weight_tensors).sum()
            aggregate_tensors = {
                key: sum(
                    entry[key] * weight
                    for entry, weight in zip(entries, weight_tensors, strict=True)
                )
                / total_weight
                for key in entries[0]
            }
            metrics = _tensor_metrics_to_scalars(aggregate_tensors)
            self._pending_diagnostics.clear()
        self._callbacks.on_epoch(*args, metrics=metrics, **kwargs)

    def _get_metrics_from_batch(self, batch: T.Tensor):
        """Compute rollout-level metrics from a full batch for logging.

        Scalar tensors are stacked for a single host transfer, and the whole
        block is skipped when the verbosity does not log diagnostics.
        """
        if not self._verbosity.logs_diagnostics:
            return {}
        metric_tensors = {
            "rollout/returns": batch["returns"].mean(),
            "rollout/advantages_mean": batch["advantages"].mean(),
            "rollout/advantages_std": batch["advantages"].std(),
            "rollout/critic_values": batch["critic_value"].mean(),
            "rollout/old_log_probs": batch["old_log_probs"].sum(-1).mean(),
        }
        action_means = batch["action"].mean((0, 1))
        action_stds = batch["action"].std((0, 1), unbiased=True)
        metric_tensors.update(
            {
                **{
                    f"rollout/action_mean_{i}": value
                    for i, value in enumerate(action_means)
                },
                **{
                    f"rollout/action_std_{i}": value
                    for i, value in enumerate(action_stds)
                },
            }
        )
        explained_variance = 1 - (batch["returns"] - batch["critic_value"]).var() / (batch["returns"].var() + self.cfg.stability.explained_variance_epsilon)
        metric_tensors["rollout/explained_variance"] = explained_variance
        return _tensor_metrics_to_scalars(metric_tensors)

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

        if self.cfg.loss.advantage_normalization_strategy == "global":
            batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (
                batch["advantages"].std() + self.cfg.stability.advantage_epsilon
            )

        self._pending_diagnostics.clear()
        self._defer_diagnostics = self._verbosity.logs_diagnostics
        try:
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
        finally:
            self._defer_diagnostics = False

        if self._scheduler:
            self._scheduler.step()
        entropy_cfg = self.cfg.loss.entropy
        self._entropy_coef = max(
            entropy_cfg.decay * self._entropy_coef,
            entropy_cfg.min_coefficient,
        )
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
            f"name={self.cfg.name!r}, "
            f"ppo_epsilon={self.cfg.loss.ppo_epsilon}, "
            f"critic_beta={self.cfg.loss.critic_beta}, "
            f"entropy_coef={self._entropy_coef}, "
            f"entropy_decay={self.cfg.loss.entropy.decay}, "
            f"target_kl={self.cfg.kl.target_kl}, "
            f"kl_warmup_steps={self.cfg.kl.warmup_steps}, "
            f"backbone_lr={self.cfg.optimizer.backbone_lr}, "
            f"cnn_lr={self.cfg.optimizer.cnn_lr or self.cfg.optimizer.backbone_lr}, "
            f"sequence_encoder_lr={self.cfg.optimizer.sequence_encoder_lr or self.cfg.optimizer.backbone_lr}, "
            f"head_lr={self.cfg.optimizer.head_lr}, "
            f"weight_decay={self.cfg.optimizer.weight_decay}, "
            f"hard_stop_kl={self.cfg.kl.hard_stop}, "
            f"advantage_normalization_strategy={self.cfg.loss.advantage_normalization_strategy!r}, "
            f"scheduler={self._scheduler.__class__.__name__ if self._scheduler else None}, "
            f"mean_reg_coef={self.cfg.loss.mean_reg_coef})"
        )

    def config(self) -> dict[str, int | float | str]:
        """Hyperparameters, suitable for mlflow.log_params (with a prefix)."""
        config: dict[str, int | float | str] = {}

        def flatten(prefix: str, value: dict | object) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    flatten(f"{prefix}.{key}" if prefix else key, nested)
            else:
                config[prefix] = "none" if value is None else value

        flatten("", self.cfg.model_dump())
        config.update({
            "entropy_coef_init": self.cfg.loss.entropy.coefficient,
            "entropy_coef_current": self._entropy_coef,
            "optimizer": self.cfg.optimizer.name,
            "scheduler_config": self.cfg.scheduler.name,
            "scheduler": self._scheduler.__class__.__name__ if self._scheduler else "none",
        })
        return config

    def network_config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable network hyperparameters."""
        return self._agent.network_config()
