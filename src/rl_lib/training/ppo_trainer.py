import numpy as np
from logging import getLogger
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Distribution
from torch.optim.lr_scheduler import LinearLR

from src.rl_lib.agent import Agent
from src.rl_lib.buffers.rollout_buffer import RolloutBuffer
from src.rl_lib.buffers.utils import to_tensor_batch
from src.rl_lib.training.callbacks.base import TrainingCallback, CallbackList


logger = getLogger(__name__)


class PPOTrainer:
    def __init__(
        self,
        agent: Agent,
        ppo_epsilon: float = 0.2,
        critic_beta: float = 0.5,
        entropy_coef: float = 0.001,
        # entropy_decay: float = 0.995,
        callbacks: list[TrainingCallback] | None = None,
    ):
        self._agent = agent
        self._optimizer = Adam(agent.network_parameters(), lr=1e-4, eps=1e-5)
        self._scheduler = LinearLR(
            self._optimizer,
            start_factor=1,
            end_factor=0.01,
            total_iters=100_000
        )
        self.ppo_epsilon = ppo_epsilon
        self.critic_beta = critic_beta
        self.entropy_coef = entropy_coef
        # self.entropy_decay = entropy_decay
        self.critic_loss_fn = nn.HuberLoss(reduction="none")
        self._callbacks = CallbackList(callbacks)
        self._step = 0

    def train_step(self, advantages: T.Tensor, returns: T.Tensor, old_log_probs: T.Tensor, old_values: T.Tensor, observation: T.Tensor, action: T.Tensor) -> dict[str, float]:
        (actor_loss, critic_loss, entropy_loss), loss_metrics = self.calculate_losses(advantages, returns, old_log_probs, old_values, observation, action)
        loss = actor_loss + self.critic_beta * critic_loss - self.entropy_coef * entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._agent.clip_grad_norm(0.5)
        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()
        self._step += 1

        metrics = {
            "loss/total": loss.detach().item(),
            "loss/actor": actor_loss.detach().item(),
            "loss/critic": critic_loss.detach().item(),
            "loss/entropy": entropy_loss.detach().item(),
        }
        return {**loss_metrics, **metrics}

    def _actor_loss(self, advantages: T.Tensor, log_probs: T.Tensor, old_log_probs: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_ratio = log_probs - old_log_probs

        ratio = log_ratio.exp()

        clipped_ratio = ratio.clamp(
            min=1 - self.ppo_epsilon,
            max=1 + self.ppo_epsilon
        )

        surrogate_loss = -T.minimum(
            ratio * normalized_advantages,
            clipped_ratio * normalized_advantages
        )

        with T.no_grad():
            approx_kl = (ratio - 1 - log_ratio).mean()

        metrics = {
            "metrics/approx_kl": approx_kl.item(),
        }
        return surrogate_loss.mean(), metrics

    def _critic_loss(self, returns: T.Tensor, values: T.Tensor, old_values: T.Tensor) -> tuple[T.Tensor, dict[str, float]]:
        clipped_values = old_values + (values - old_values).clamp(-self.ppo_epsilon, self.ppo_epsilon)
        loss_unclipped = self.critic_loss_fn(values, returns)
        loss_clipped = self.critic_loss_fn(clipped_values, returns)
        return T.maximum(loss_unclipped, loss_clipped).mean(), {}

    def _entropy_loss(self, dist: Distribution) -> tuple[T.Tensor, dict[str, float]]:
        # entropy of the base Normal; TanhTransform doesn't have closed-form entropy
        entropy: T.Tensor = dist.base_dist.entropy()
        mean_entropy = entropy.mean(0).detach()
        metrics = {
            f"metrics/entropy_{i}": value.item() for i, value in enumerate(mean_entropy)
        }
        return entropy.sum(dim=-1).mean(), metrics


    def calculate_losses(
        self,
        advantages: T.Tensor,
        returns: T.Tensor,
        old_log_probs: T.Tensor,
        old_values: T.Tensor,
        observation: T.Tensor,
        action: T.Tensor
    ) -> tuple[tuple[T.Tensor, T.Tensor, T.Tensor], dict[str, float]]:
        log_probs, values, dist = self._agent.evaluate_actions(observation, action)
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

    def train(self, buffer: RolloutBuffer, epochs: int, minibatch_size: int, rng: np.random.Generator):
        stack_size = self._agent._stack_size
        self._on_start(step=self._step, metrics=buffer.get_metrics(stack_size))
        for epoch in range(epochs):
            for minibatch in buffer.get_iid_minibatches(minibatch_size, stack_size, rng):
                tensor_batch = to_tensor_batch(minibatch, self._agent._device)
                metrics = self.train_step(**tensor_batch)
                self._on_minibatch(metrics=metrics, step=self._step)
            self._on_epoch()

        self.entropy_coef = max(self.entropy_coef - 1e-5, 1e-5)
        metrics = {
            "training/entropy_coef": self.entropy_coef,
            "training/lr": self._optimizer.param_groups[0]["lr"]
        }
        self._on_end(metrics=metrics, step=self._step)
        buffer.reset()
