"""Rollout buffer with GAE computation for PPO.

Stores transitions from vectorized environments and computes returns/advantages
using Generalized Advantage Estimation. Handles the shift between stored done
flags (marking step *end*) and returned dones (marking frame *start* of new
episode) to align with Agent's temporal masking convention.
"""

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
    """Preallocated circular storage for PPO rollouts with GAE.

    Allocates field tensors on the first step and copies subsequent transitions
    into circular slots. When full, computes returns and advantages using GAE
    with proper handling of termination vs truncation (bootstrap through
    truncation, stop at either).

    The buffer stores done flags as produced by env.step() (marking the step
    that *ended* an episode), but get() returns dones shifted by one column
    so they mean "this frame is the first of a new episode" — the convention
    Agent._get_mask_window uses while collecting.
    """

    def __init__(
        self,
        config: RolloutSettings,
        stack_size: int,
    ) -> None:
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
        self._buffer: dict[str, T.Tensor] = {}
        self._transition_index = 0
        self._window_index = 0
        self._transition_count = 0
        self._window_count = 0
        self._counter: int = 0
        self.reset()

    def is_full(self) -> bool:
        """Check if buffer has collected enough steps for an update."""
        return self._counter == self.size

    def _allocate_storage(self, step: RolloutStep) -> None:
        """Allocate all rollout tensors from the first step's tensor metadata."""
        values = {
            "observation": step.observation,
            "action": step.action,
            "critic_value": step.critic_value,
            "old_log_probs": step.old_log_probs,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "truncated_value": (
                T.zeros_like(step.reward)
                if step.truncated_value is None
                else step.truncated_value
            ),
        }
        window_capacity = self.size + self._stack_size - 1
        window_fields = {"observation", "terminated", "truncated"}
        self._buffer = {
            key: value.new_empty(
                (
                    window_capacity if key in window_fields else self.size,
                    *value.shape,
                )
            )
            for key, value in values.items()
        }

    def _write(self, key: str, value: T.Tensor, index: int) -> None:
        """Copy a step tensor into its preallocated circular slot."""
        storage = self._buffer[key]
        if value.shape != storage.shape[1:]:
            raise ValueError(
                f"{key} shape changed from {tuple(storage.shape[1:])} "
                f"to {tuple(value.shape)}"
            )
        if value.dtype != storage.dtype or value.device != storage.device:
            raise ValueError(
                f"{key} dtype/device changed from {storage.dtype}/{storage.device} "
                f"to {value.dtype}/{value.device}"
            )
        storage[index].copy_(value)

    def _ordered(self, key: str, windowed: bool = False) -> T.Tensor:
        """Return chronological storage, concatenating only when it wraps."""
        if windowed:
            count = self._window_count
            cursor = self._window_index
            capacity = self.size + self._stack_size - 1
        else:
            count = self._transition_count
            cursor = self._transition_index
            capacity = self.size

        start = (cursor - count) % capacity
        end = start + count
        storage = self._buffer[key]
        if end <= capacity:
            ordered = storage[start:end]
        else:
            ordered = T.cat((storage[start:], storage[: end - capacity]), dim=0)
        return ordered.movedim(0, 1)

    def _append_window_values(self, values: dict[str, T.Tensor]) -> None:
        """Write one observation-aligned column and advance its shared cursor."""
        for key in ("observation", "terminated", "truncated"):
            self._write(key, values[key], self._window_index)
        self._window_index = (self._window_index + 1) % (
            self.size + self._stack_size - 1
        )
        self._window_count = min(
            self._window_count + 1, self.size + self._stack_size - 1
        )

    def add(self, step: RolloutStep) -> None:
        """Copy one vectorized environment step into the rollout storage.

        Args:
            step: RolloutStep containing all transition data for one env step.
        """
        if not self._buffer:
            self._allocate_storage(step)

        values = {
            "observation": step.observation,
            "action": step.action,
            "critic_value": step.critic_value,
            "old_log_probs": step.old_log_probs,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "truncated_value": (
                T.zeros_like(step.reward)
                if step.truncated_value is None
                else step.truncated_value
            ),
        }

        if self._window_count == 0:
            for _ in range(self._stack_size - 1):
                self._append_window_values(values)

        self._append_window_values(values)
        for key in (
            "action",
            "critic_value",
            "old_log_probs",
            "reward",
            "truncated_value",
        ):
            self._write(key, values[key], self._transition_index)

        self._transition_index = (self._transition_index + 1) % self.size
        self._transition_count = min(self._transition_count + 1, self.size)

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
        decay = (
            gamma
            * gae_lambda
            * T.logical_not(dones).to(dtype=reward.dtype)
        )
        advantages = delta
        offset = 1
        while offset < reward.shape[1]:
            shifted_advantages = T.cat(
                (advantages[:, offset:], T.zeros_like(advantages[:, :offset])),
                dim=1,
            )
            shifted_decay = T.cat(
                (decay[:, offset:], T.zeros_like(decay[:, :offset])),
                dim=1,
            )
            advantages = advantages + decay * shifted_advantages
            decay = decay * shifted_decay
            offset *= 2
        returns = advantages + critic_value[:, :-1]
        return returns, advantages

    def get(self, gamma: float | None = None, gae_lambda: float | None = None) -> dict[str, T.Tensor]:
        """Compute returns/advantages and return flattened batch for PPO update.

        Returns tensors with batch dimension (num_envs) and a sequence dimension
        for minibatch sampling. The done flags are shifted so they
        indicate "this frame starts a new episode". Stored rollout fields may
        be views into the circular storage; callers should treat them as
        read-only. Returns and advantages are newly computed tensors.

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
        if not self._buffer:
            raise ValueError("cannot compute a batch from an empty rollout buffer")
        buffer = {
            key: self._ordered(key, windowed=key in {"observation", "terminated", "truncated"})
            for key in self._buffer
        }
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

    def reset_counter(self) -> None:
        """Reset step counter without clearing buffer (for multi-epoch updates)."""
        self._counter = 0

    def reset(self) -> None:
        """Clear buffer and reset counter.

        Tensors are allocated lazily on the next ``add``. Observation and done
        fields reserve ``size + stack_size - 1`` slots for initial padding.
        """
        self._buffer = {}
        self._transition_index = 0
        self._window_index = 0
        self._transition_count = 0
        self._window_count = 0
        self.reset_counter()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"stack_size={self._stack_size}, "
            f"counter={self._counter}/{self.size}, "
            f"is_full={self.is_full()})"
        )

    def config(self) -> dict[str, int | float | str]:
        """Hyperparameters, suitable for mlflow.log_params (with a prefix)."""
        return {
            "size": self.size,
            "stack_size": self._stack_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
        }
