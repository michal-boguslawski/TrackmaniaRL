from collections import deque
from dataclasses import dataclass
import numpy as np
import torch as T
from math import ceil


@dataclass(slots=True, frozen=True)
class RolloutStep:
    observation: T.Tensor
    action: T.Tensor
    critic_value: T.Tensor
    old_log_probs: T.Tensor
    reward: T.Tensor
    terminated: T.Tensor
    truncated: T.Tensor


# Rollout Buffer
class RolloutBuffer:
    def __init__(
        self,
        size: int,
        # num_envs: int,
        stack_size: int,
        # observation_space: tuple[int, ...],
        # action_space: tuple[int, ...],
        # obs_dtype: T.dtype = T.uint8,
        # device: str = "cpu",
        *args,
        **kwargs
    ):
        self.size = size
        # self._num_envs = num_envs
        # self._observation_space = observation_space
        # self._action_space = action_space
        # self._obs_dtype = obs_dtype
        self._stack_size = stack_size
        # self._device = T.device(device)
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
            self._append_to_buffer(key, getattr(step, key), if_append_start)

        self._counter += 1

    @staticmethod
    def compute_returns_and_advantages(
        reward: T.Tensor,
        critic_value: T.Tensor,
        terminated: T.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[T.Tensor, T.Tensor]:
        """
        reward shape is (batch, length - 1)
        critic_value shape is (batch, length)
        terminated shape is (batch, length - 1)
        """
        
        delta = reward + gamma * critic_value[:, 1:] * T.logical_not(terminated) - critic_value[:, :-1]
        advantages = T.zeros_like(reward)
        returns = T.zeros_like(reward)
        last_gae_lam = 0.
        for i in reversed(range(1, reward.shape[1])):
            last_gae_lam = delta[:, i] + gamma * gae_lambda * last_gae_lam * T.logical_not(terminated[:, i])
            advantages[:, i] = last_gae_lam
        returns = advantages + critic_value[:, :1]
        return returns, advantages

    def get(self, gamma: float = 0.99, gae_lambda: float = 0.97) -> dict[str, T.Tensor]:
        buffer = {key: T.stack(list(value), dim=1) for key, value in self._buffer.items()}
        returns, advantages = self.compute_returns_and_advantages(
            buffer["reward"][:, :-1],
            buffer["critic_value"],
            buffer["terminated"][:, (self._stack_size-1):-1],
            gamma,
            gae_lambda,
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
        }

        self.reset_counter()


if __name__ == "__main__":
    buffer = RolloutBuffer(32, 4)
    for i in range(1, 37):
        step = RolloutStep(
            T.tensor([[i],[i+1]]).to(T.float32),
            T.tensor([i, i+1]).to(T.float32),
            T.tensor([i, i+1]).to(T.float32),
            T.tensor([i, i+1]).to(T.float32),
            T.tensor([i, i+1]).to(T.float32),
            T.tensor([False, False]),
            T.tensor([False, False]),
        )
        buffer.add(step)
    batch = buffer.get()
    print({key: value.shape for key, value in batch.items()})
    actions = batch["action"]
    num_envs, batch_size = actions.shape
    indices = np.random.permutation(ceil(batch_size * num_envs / 8))
    print(indices * 8)
    for ind in indices:
        i, k = ind % num_envs, ind // num_envs
        print(i, k)
    print(indices[0])
    # print(
    #     batch["observation"][i, k:(k+32)]
    # )
    # print(buffer.is_full())
    # buffer.reset()
    # print(buffer._buffer)

