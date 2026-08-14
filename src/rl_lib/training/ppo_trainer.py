import numpy as np
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Distribution

from src.rl_lib.agent import Agent
from src.rl_lib.buffers.rollout_buffer import RolloutBuffer
from src.rl_lib.buffers.utils import to_tensor_batch
from src.rl_lib.training.callbacks.base import TrainingCallback, CallbackList


class PPOTrainer:
    def __init__(
        self,
        agent: Agent,
        ppo_epsilon: float = 0.2,
        critic_beta: float = 0.5,
        entropy_coef: float = 0.001,
        callbacks: list[TrainingCallback] | None = None,
    ):
        self._agent = agent
        self._optimizer = Adam(agent.network_parameters(), lr=3e-4, eps=1e-5)
        self.ppo_epsilon = ppo_epsilon
        self.critic_beta = critic_beta
        self.entropy_coef = entropy_coef
        self.critic_loss_fn = nn.MSELoss()
        self._callbacks = CallbackList(callbacks)
        self._step = 0

    def train_step(self, advantages: T.Tensor, returns: T.Tensor, old_log_probs: T.Tensor, observation: T.Tensor, action: T.Tensor) -> dict[str, float]:
        actor_loss, critic_loss, entropy_loss = self.calculate_losses(advantages, returns, old_log_probs, observation, action)
        loss = actor_loss + self.critic_beta * critic_loss - self.entropy_coef * entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._agent.clip_grad_norm(0.5)
        self._optimizer.step()
        self._step += 1

        return {
            "loss/total": loss.detach().item(),
            "loss/actor": actor_loss.detach().item(),
            "loss/critic": critic_loss.detach().item(),
            "loss/entropy": entropy_loss.detach().item(),
        }

    def _actor_loss(self, advantages: T.Tensor, log_probs: T.Tensor, old_log_probs: T.Tensor) -> T.Tensor:
        log_ratio = (log_probs - old_log_probs).clamp(-10, 10)  # prevent exp overflow
        ratio = log_ratio.exp_()
        # ratio = (log_probs - old_log_probs).exp_()
        clipped_ratio = ratio.clamp(min=1 - self.ppo_epsilon, max=1 + self.ppo_epsilon)
        surrogate_loss = -T.minimum(ratio * advantages, clipped_ratio * advantages)
        return surrogate_loss.mean()

    def _critic_loss(self, returns: T.Tensor, values: T.Tensor) -> T.Tensor:
        critic_loss = self.critic_loss_fn(returns, values)
        return critic_loss

    # def _entropy_loss(self, log_probs: T.Tensor) -> T.Tensor:
    #     return -log_probs.mean()
    def _entropy_loss(self, dist: Distribution) -> T.Tensor:
        # entropy of the base Normal; TanhTransform doesn't have closed-form entropy
        return dist.base_dist.entropy().sum(dim=-1).mean()


    def calculate_losses(self, advantages: T.Tensor, returns: T.Tensor, old_log_probs: T.Tensor, observation: T.Tensor, action: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        log_probs, values, dist = self._agent.evaluate_actions(observation, action)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        actor_loss = self._actor_loss(normalized_advantages, log_probs, old_log_probs)
        critic_loss = self._critic_loss(returns, values)
        entropy_loss = self._entropy_loss(dist)
        return actor_loss, critic_loss, entropy_loss

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
        self._on_end()
        buffer.reset()
