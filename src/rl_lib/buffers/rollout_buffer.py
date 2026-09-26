from collections import deque
from dataclasses import dataclass
import torch as T
from rl_lib.run_config import RolloutSettings


@dataclass(slots=True, frozen=True)
class RolloutStep:
    observation: T.Tensor
    action: T.Tensor
    critic_value: T.Tensor
    old_log_probs: T.Tensor
    reward: T.Tensor
    terminated: T.Tensor
    truncated: T.Tensor
    truncated_value: T.Tensor | None = None


# Rollout Buffer
class RolloutBuffer:
    def __init__(
        self,
        size: int,
        stack_size: int,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        *args,
        **kwargs
    ):
        self.size = size
        self._stack_size = stack_size
        defaults = RolloutSettings()
        self.gamma = defaults.gamma if gamma is None else gamma
        self.gae_lambda = defaults.gae_lambda if gae_lambda is None else gae_lambda
        self._buffer: dict[str, deque[T.Tensor]] = {}
        self._counter: int = 0
        self.reset()

    def is_full(self) -> bool:
        return self._counter == self.size

    def _append_to_buffer(self, key: str, value: T.Tensor, if_append_start: bool = False):
        if if_append_start and len(self._buffer[key]) < self._stack_size - 1:
            for _ in range(self._stack_size - 1):
                self._buffer[key].append(value.clone())

        if key not in self._buffer:
            self._buffer[key] = deque(maxlen=self.size)
        self._buffer[key].append(value.clone())

    def add(self, step: RolloutStep):
        # if self._counter >= self.size:
        #     raise IndexError("Rollout buffer is full")

        for key in self._buffer.keys():
            if_append_start = (key in ["observation", "terminated", "truncated"])
            value = getattr(step, key)
            if value is None:
                value = T.zeros_like(step.reward)
            self._append_to_buffer(key, value, if_append_start)

        self._counter += 1

    @staticmethod
    def compute_returns_and_advantages(
        reward: T.Tensor,
        critic_value: T.Tensor,
        terminated: T.Tensor,
        truncated: T.Tensor,
        gamma: float,
        gae_lambda: float,
        truncated_value: T.Tensor | None = None,
    ) -> tuple[T.Tensor, T.Tensor]:
        """
        reward shape is (batch, length - 1)
        critic_value shape is (batch, length)
        terminated shape is (batch, length - 1)
        """
        dones = T.logical_or(terminated, truncated)
        next_value = critic_value[:, 1:]
        if truncated_value is not None:
            next_value = T.where(truncated, truncated_value, next_value)
        delta = (
            reward
            + gamma * next_value * T.logical_not(terminated)
            - critic_value[:, :-1]
        )
        advantages = T.zeros_like(reward)
        last_gae_lam = 0.
        for i in reversed(range(reward.shape[1])):
            last_gae_lam = delta[:, i] + gamma * gae_lambda * last_gae_lam * T.logical_not(dones[:, i])
            advantages[:, i] = last_gae_lam
        returns = advantages + critic_value[:, :-1]
        return returns, advantages

    def get(self, gamma: float | None = None, gae_lambda: float | None = None) -> dict[str, T.Tensor]:
        gamma = self.gamma if gamma is None else gamma
        gae_lambda = self.gae_lambda if gae_lambda is None else gae_lambda
        buffer = {key: T.stack(list(value), dim=1) for key, value in self._buffer.items()}
        returns, advantages = self.compute_returns_and_advantages(
            buffer["reward"][:, :-1],
            buffer["critic_value"],
            buffer["terminated"][:, (self._stack_size-1):-1],
            buffer["truncated"][:, (self._stack_size-1):-1],
            gamma,
            gae_lambda,
            buffer["truncated_value"][:, :-1],
        )
        dones = T.logical_or(
            buffer["terminated"],
            buffer["truncated"]
        )

        flat = {
            "observation": buffer["observation"][:, :-1],  # (batch, length + stack_size - 2, *obs_shape)
            "action": buffer["action"][:, :-1],  # (batch, length-1, *action_shape)
            "old_log_probs": buffer["old_log_probs"][:, :-1],  # (batch, length-1)
            "critic_value": buffer["critic_value"][:, :-1],  # (batch, length-1)
            "returns": returns,  # (batch, length - 1)
            "advantages": advantages,  # (batch, length - 1)
            "dones": dones[:, :-1],  # (batch, length + stack_size - 2)
        }
        return flat

    def reset_counter(self):
        self._counter = 0

    def reset(self):
        self._buffer = {
            "observation": deque(maxlen=self.size + self._stack_size - 1),
            "action": deque(maxlen=self.size),
            "critic_value": deque(maxlen=self.size),
            "old_log_probs": deque(maxlen=self.size),
            "reward": deque(maxlen=self.size),
            "truncated": deque(maxlen=self.size + self._stack_size - 1),
            "terminated": deque(maxlen=self.size + self._stack_size - 1),
            "truncated_value": deque(maxlen=self.size),
        }

        self.reset_counter()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"stack_size={self._stack_size}, "
            f"counter={self._counter}/{self.size}, "
            f"is_full={self.is_full()})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters, suitable for mlflow.log_params (with a prefix)."""
        return {
            "size": self.size,
            "stack_size": self._stack_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
        }
