from math import ceil
import numpy as np
from numpy.typing import NDArray
from logging import getLogger
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Distribution
from torch.optim.lr_scheduler import LinearLR
from torch.nn.modules.loss import _Loss
from typing import Iterator, Literal

from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.buffers.utils import to_tensor_batch
from rl_lib.training.callbacks.base import TrainingCallback, CallbackList


logger = getLogger(__name__)


class PPOTrainer:
    def __init__(
        self,
        agent: Agent,
        ppo_epsilon: float = 0.2,
        critic_beta: float = 0.5,
        entropy_coef: float = 0.001,
        advantage_normalization_strategy: Literal["batch", "global"] | None = None,
        # entropy_decay: float = 0.995,
        callbacks: list[TrainingCallback] | None = None,
    ):
        self._agent = agent
        self._optimizer = Adam(
            agent.network_parameters(),
            # lr=3e-4,
            eps=1e-5,
        )
        self._scheduler = LinearLR(
            self._optimizer,
            start_factor=1,
            end_factor=0.01,
            total_iters=400_000
        )
        self.ppo_epsilon = ppo_epsilon
        self.critic_beta = critic_beta
        self.entropy_coef = entropy_coef
        self.advantage_normalization_strategy = advantage_normalization_strategy
        # self.entropy_decay = entropy_decay
        self._critic_loss_fn = nn.HuberLoss(reduction="none")
        self._callbacks = CallbackList(callbacks)
        self._step = 0
        self.target_kl = 0.1
        self._agent.train()

    @property
    def stack_size(self) -> int:
        return self._agent.stack_size

    @property
    def device(self) -> T.device:
        return self._agent.device

    def act(self, observation: T.Tensor, done: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        action, log_probs, critic_value = self._agent.act(observation, done)
        return action, log_probs, critic_value

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

        (actor_loss, critic_loss, entropy_loss), loss_metrics = self.calculate_losses(advantages, returns, old_log_probs, old_values, observation, action, dones)
        loss = actor_loss + self.critic_beta * critic_loss - self.entropy_coef * entropy_loss

        self._optimizer.zero_grad()
        loss.backward()

        metrics = self._get_train_step_metrics(loss)
        grad_norm = self._agent.clip_grad_norm(0.5)

        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()
        self._step += 1

        return {**loss_metrics, **metrics}

    def _get_train_step_metrics(self, loss: T.Tensor) -> dict[str, float]:
        with T.no_grad():
            metrics = {
                "loss/total": loss.detach().item(),
                **self._agent.get_parital_clip_grad_norms()
            }
        return metrics

    def _actor_loss(self, advantages: T.Tensor, log_probs: T.Tensor, old_log_probs: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        if self.advantage_normalization_strategy and self.advantage_normalization_strategy == "batch":
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        assert log_probs.shape == old_log_probs.shape
        log_ratio = log_probs - old_log_probs

        ratio = log_ratio.sum(-1).clamp(-20, 20).exp()

        clipped_ratio = ratio.clamp(
            min=1 - self.ppo_epsilon,
            max=1 + self.ppo_epsilon
        )

        assert advantages.shape == ratio.shape
        surrogate_loss = -T.minimum(
            ratio * advantages,
            clipped_ratio * advantages
        )
        actor_loss = surrogate_loss.mean()

        metrics = self._actor_loss_metrics(log_ratio)
        metrics["metrics/actor_loss"] = actor_loss.detach().item()

        return actor_loss, metrics

    def _actor_loss_metrics(self, log_ratio: T.Tensor) -> dict[str, float]:
        with T.no_grad():
            log_ratio_total = log_ratio.sum(-1)
            ratio_total = log_ratio_total.exp()
            approx_kl = (ratio_total - 1 - log_ratio_total).mean()
            ratio_per_action = log_ratio.exp()
            approx_kl_per_action = (ratio_per_action - 1 - log_ratio).mean(dim=0)
            ratio_max = ratio_total.max()
            clip_fraction = (T.abs(ratio_total - 1) > self.ppo_epsilon).float().mean()

        metrics = {
            "metrics/approx_kl": approx_kl.item(),
            "metrics/ratio_max": ratio_max.item(),
            "metrics/clip_fraction": clip_fraction.item(),
        }
        
        for i, value in enumerate(approx_kl_per_action):
            metrics[f"metrics/approx_kl_{i}"] = value.item()

        return metrics

    def _critic_loss(self, returns: T.Tensor, values: T.Tensor, old_values: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        assert values.shape == old_values.shape == returns.shape
        clipped_values = old_values + (values - old_values).clamp(-self.ppo_epsilon, self.ppo_epsilon)
        loss_unclipped = self._critic_loss_fn(values, returns)
        loss_clipped = self._critic_loss_fn(clipped_values, returns)
        
        critic_loss = T.maximum(loss_unclipped, loss_clipped).mean()

        metrics = {
            "loss/critic": critic_loss.detach().item(),
        }
        return critic_loss, metrics

    def _entropy_loss(self, dist: Distribution) -> tuple[T.Tensor, dict[str, float]]:
        # entropy of the base Normal; TanhTransform doesn't have closed-form entropy
        entropy: T.Tensor = dist.base_dist.entropy()
        entropy_loss = entropy.sum(dim=-1).mean()

        mean_entropy = entropy.mean(0).detach()
        metrics = {
            f"metrics/entropy_{i}": value.item() for i, value in enumerate(mean_entropy)
        }
        metrics["loss/entropy"] = entropy_loss.detach().item()

        with T.no_grad():
            log_std = dist.base_dist.scale.log().mean(0)
        metrics.update({f"metrics/log_std_{i}": v.item() for i, v in enumerate(log_std)})

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
        log_probs, values, dist = self._agent.evaluate_actions(observation, action, dones)
        actor_loss, actor_metrics = self._actor_loss(advantages, log_probs, old_log_probs)
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
    def _get_iid_minibatches(batch: dict[str, T.Tensor], minibatch_size: int, stack_size: int) -> Iterator[dict[str, T.Tensor]]:
        """Iterate over IID minibatches from a batch of sequences."""

        num_envs, batch_size, _ = batch["action"].shape
        indices = np.random.permutation(batch_size * num_envs)
        for index in range(0, batch_size * num_envs, minibatch_size):
            subindices = [(ind % num_envs, ind // num_envs) for ind in indices[index:(index+minibatch_size)]]
            
            
            yield {
                "observation": T.cat(
                    [batch["observation"][i, k:(k+stack_size)] for i, k in subindices],
                    dim=0,
                ),
                "action": T.stack([batch["action"][i, k] for i, k in subindices], dim=0),
                "old_log_probs": T.stack([batch["old_log_probs"][i, k] for i, k in subindices], dim=0),
                "old_values": T.stack([batch["critic_value"][i, k] for i, k in subindices], dim=0),
                "returns": T.stack([batch["returns"][i, k] for i, k in subindices], dim=0),
                "advantages": T.stack([batch["advantages"][i, k] for i, k in subindices], dim=0),
                "dones": T.cat(
                    [batch["dones"][i, k:(k+stack_size)] for i, k in subindices],
                    dim=0,
                ),
            }

    @staticmethod
    def _get_metrics_from_batch(batch: T.Tensor):
        metrics = {
            "rollout/returns": batch["returns"].mean().item(),
            "rollout/advantages_mean": batch["advantages"].mean().item(),
            "rollout/advantages_std": batch["advantages"].std().item(),
            "rollout/critic_values": batch["critic_value"].mean().item(),
            "rollout/old_log_probs" : batch["old_log_probs"].mean().item(),
        }
        action_means = batch["action"].mean((0, 1))
        for i, mean in enumerate(action_means):
            metrics[f"rollout/action_mean_{i}"] = mean.item()
        action_stds = batch["action"].std((0, 1), unbiased=True)
        for i, std in enumerate(action_stds):
            metrics[f"rollout/action_std_{i}"] = std.item()

        explained_variance = 1 - (batch["returns"] - batch["critic_value"]).var() / (batch["returns"].var() + 1e-8)
        metrics["rollout/explained_variance"] = explained_variance
        return metrics

    def train(
        self,
        batch: dict[str, T.Tensor],
        epochs: int,
        minibatch_size: int,
        training_step: int,
        # rng: np.random.Generator,
    ):

        self._on_start(
            step=training_step,
            metrics=self._get_metrics_from_batch(batch)
        )

        if self.advantage_normalization_strategy and self.advantage_normalization_strategy == "global":
            batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-4)

        for epoch in range(epochs):
            epoch_kls = []
            for minibatch in self._get_iid_minibatches(batch, minibatch_size, self.stack_size):
                try:
                    minibatch_metrics = self.train_step(**minibatch)
                except Exception as e:
                    logger.error(f"Error at training step {training_step}, epoch {epoch}: {e}")
                    raise e
                epoch_kls.append(minibatch_metrics["metrics/approx_kl"])
                self._on_minibatch(metrics=minibatch_metrics, step=self._step)
            mean_epoch_kl = float(np.mean(epoch_kls))
            self._on_epoch()
            if mean_epoch_kl > self.target_kl:
                logger.warning(f"Early stop epoch {epoch}: KL {mean_epoch_kl:.4f} > {self.target_kl}")
                break

        # self.entropy_coef = max(self.entropy_coef - 1e-5, 1e-5)
        metrics = {
            "training/entropy_coef": self.entropy_coef,
            # "training/lr": self._optimizer.param_groups[0]["lr"]
        }
        for i, pg in enumerate(self._optimizer.param_groups):
            metrics[f"training/lr_{i}"] = pg["lr"]
        self._on_end(metrics=metrics, step=training_step)
