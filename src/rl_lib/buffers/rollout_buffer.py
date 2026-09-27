"""Rollout buffer with GAE computation for PPO.

Stores transitions from vectorized environments and computes returns/advantages
using Generalized Advantage Estimation. Handles the shift between stored done
flags (marking step *end*) and returned dones (marking frame *start* of new
episode) to align with Agent's temporal masking convention.
"""

from collections import deque
from dataclasses import dataclass
import torch as T
from rl_lib.run_config import RolloutSettings


@dataclass(slots=True, frozen=True)
class RolloutStep:
    """Single transition stored in the rollout buffer.

    Attributes:
        observation: Raw observation (NHWC, uint8) of shape (num_envs, H, W, C).
        action: Action taken, shape (num_envs, action_dim).
        critic_value: Value estimate V(s), shape (num_envs,).
        old_log_probs: Log-prob of action under old policy, shape (num_envs,).
        reward: Scalar reward, shape (num_envs,).
        terminated: Boolean, true if episode terminated at this step.
        truncated: Boolean, true if episode truncated (time limit) at this step.
        truncated_value: Bootstrap value for truncated episodes, shape (num_envs,)
            or None if no truncation occurred.
    """
    observation: T.Tensor
    action: T.Tensor
    critic_value: T.Tensor
    old_log_probs: T.Tensor
    reward: T.Tensor
    terminated: T.Tensor
    truncated: T.Tensor
    truncated_value: T.Tensor | None = None


class RolloutBuffer:
    """Fixed-size circular buffer for PPO rollouts with GAE.

    Stores transitions from multiple vectorized environments. When full,
    computes returns and advantages using GAE with proper handling of
    termination vs truncation (bootstrap through truncation, stop at either).

    The buffer stores done flags as produced by env.step() (marking the step
    that *ended* an episode), but get() returns dones shifted by one column
    so they mean "this frame is the first of a new episode" — the convention
    Agent._get_mask_window uses while collecting.
    """

    def __init__(
        self,
        config: RolloutSettings,
        stack_size: int,
    ):
        """Initialize the rollout buffer.

        Args:
            config: RolloutSettings with buffer_size, gamma, gae_lambda.
            stack_size: Number of frames in temporal window (for obs/done padding).
        """
        self.cfg = config
        self.size = config.buffer_size
        self._stack_size = stack_size
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self._buffer: dict[str, deque[T.Tensor]] = {}
        self._counter: int = 0
        self.reset()

    def is_full(self) -> bool:
        """Check if buffer has collected enough steps for an update."""
        return self._counter == self.size

    def _append_to_buffer(self, key: str, value: T.Tensor, if_append_start: bool = False):
        """Append tensor to buffer, optionally pre-filling with copies for padding.

        Args:
            key: Buffer key (e.g., "observation", "reward").
            value: Tensor to append (will be cloned).
            if_append_start: If True and buffer has fewer than stack_size-1
                entries, pre-fill with copies of value. Used for observation
                and done buffers to provide history for first steps.
        """
        if if_append_start and len(self._buffer[key]) < self._stack_size - 1:
            for _ in range(self._stack_size - 1):
                self._buffer[key].append(value.clone())

        if key not in self._buffer:
            self._buffer[key] = deque(maxlen=self.size)
        self._buffer[key].append(value.clone())

    def add(self, step: RolloutStep):
        """Add a rollout step to the buffer.

        Args:
            step: RolloutStep containing all transition data for one env step.
        """
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
        """Compute GAE returns and advantages.

        Bootstraps through truncation (uses truncated_value) but stops
        advantage recursion at either termination or truncation.

        Args:
            reward: Tensor of shape (batch, length - 1).
            critic_value: Tensor of shape (batch, length).
            terminated: Boolean tensor of shape (batch, length - 1).
            truncated: Boolean tensor of shape (batch, length - 1).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            truncated_value: Optional bootstrap values for truncated steps,
                shape (batch, length - 1).

        Returns:
            Tuple of (returns, advantages), each shape (batch, length - 1).
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
        """Compute returns/advantages and return flattened batch for PPO update.

        Returns tensors with batch dimension (num_envs) and sequence dimension
        flattened for minibatch sampling. The done flags are shifted so they
        indicate "this frame starts a new episode".

        Args:
            gamma: Override discount factor (default: config.gamma).
            gae_lambda: Override GAE lambda (default: config.gae_lambda).

        Returns:
            Dict with keys:
            - observation: (batch, length + stack_size - 2, H, W, C)
            - action: (batch, length - 1, action_dim)
            - old_log_probs: (batch, length - 1)
            - critic_value: (batch, length - 1)
            - returns: (batch, length - 1)
            - advantages: (batch, length - 1)
            - dones: (batch, length + stack_size - 2) — shifted flags
        """
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
        episode_ends = T.logical_or(
            buffer["terminated"],
            buffer["truncated"]
        )
        # `dones` has to answer "is this frame the first of a new episode", the
        # question `Agent._get_mask_window` asks while collecting: a frame is
        # history only if no episode boundary precedes it inside the window. The
        # stored flags answer a different question - "did stepping at this frame
        # end the episode" - and that boundary belongs to the *next* frame. So
        # the flags move one column to the right, and the leading columns that
        # `observation` fills by duplicating step 0 start no episode.
        dones = T.cat(
            [
                T.zeros_like(episode_ends[:, : self._stack_size]),
                episode_ends[:, self._stack_size - 1 : -1],
            ],
            dim=1,
        )

        flat = {
            "observation": buffer["observation"][:, :-1],
            "action": buffer["action"][:, :-1],
            "old_log_probs": buffer["old_log_probs"][:, :-1],
            "critic_value": buffer["critic_value"][:, :-1],
            "returns": returns,
            "advantages": advantages,
            "dones": dones[:, :-1],
        }
        return flat

    def reset_counter(self):
        """Reset step counter without clearing buffer (for multi-epoch updates)."""
        self._counter = 0

    def reset(self):
        """Clear buffer and reset counter.

        Observation and done buffers have maxlen = size + stack_size - 1
        to accommodate the initial padding copies.
        """
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
