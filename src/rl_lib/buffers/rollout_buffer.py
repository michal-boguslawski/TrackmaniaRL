from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Iterator


@dataclass(slots=True, frozen=True)
class RolloutStep:
    observation: NDArray[np.uint8]
    action: NDArray[np.float32]
    critic_value: NDArray[np.float32]
    old_log_probs: NDArray[np.float32]
    reward: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]


# Rollout Buffer
class RolloutBuffer:
    def __init__(
        self,
        size: int,
        num_envs: int,
        observation_space: tuple[int, ...],
        action_space: tuple[int, ...],
        obs_dtype: np.dtype = np.dtype(np.uint8)
    ):
        self.size = size
        self._num_envs = num_envs
        self._observation_space = observation_space
        self._action_space = action_space
        self._obs_dtype = obs_dtype
        self._buffer: dict[str, NDArray] = {}
        self._counter: int = 0
        self._cached_flat: dict[str, NDArray] | None = None
        self._cache_key: tuple[int, float, float] | None = None
        self.reset()

    def is_full(self) -> bool:
        return self._counter == self.size

    def add(self, step: RolloutStep):
        i = self._counter
        self._buffer["observation"][:, i] = step.observation
        self._buffer["action"][:, i] = step.action
        self._buffer["critic_value"][:, i] = step.critic_value
        self._buffer["old_log_probs"][:, i] = step.old_log_probs
        self._buffer["reward"][:, i] = step.reward
        self._buffer["terminated"][:, i] = step.terminated
        self._buffer["truncated"][:, i] = step.truncated
        self._counter += 1
        self._cached_flat = None

    @staticmethod
    def compute_returns_and_advantages(
        reward: NDArray[np.float32],
        critic_value: NDArray[np.float32],
        done: NDArray[np.bool_],
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        delta = reward[:, :-1] + gamma * critic_value[:, 1:] * np.logical_not(done[:, :-1]) - critic_value[:, :-1]
        advantages = np.zeros_like(delta)
        last_gae_lam = 0
        for i in reversed(range(advantages.shape[1])):
            last_gae_lam = delta[:, i] + gamma * gae_lambda * last_gae_lam * np.logical_not(done[:, i])
            advantages[:, i] = last_gae_lam
        returns = advantages + critic_value[:, :-1]
        return returns, advantages

    def _build_observation_windows(self, obs: NDArray[np.uint8], done: NDArray[np.bool_], stack_size: int) -> NDArray[np.uint8]:
        """
        Returns shape (size, num_envs, stack_size, *observation_space).
        stacked[n, t] = last `stack_size` frames ending at t, for env n,
        clipped at episode boundaries (terminated/truncated) and buffer start,
        padded by repeating the earliest available frame.
        """
        T_, N, k = self.size, self._num_envs, stack_size

        stacked = np.zeros((N, T_, k) + self._observation_space, dtype=self._obs_dtype)
        for n in range(N):
            window_start = 0
            for t in range(T_):
                if t > 0 and done[n, t - 1]:
                    window_start = t
                lo = max(window_start, t - k + 1)
                frames = obs[n, lo:t + 1]
                pad = k - frames.shape[0]
                if pad > 0:
                    frames = np.concatenate([np.repeat(frames[:1], pad, axis=0), frames], axis=0)
                stacked[n, t] = frames
        return stacked

    def get_flat_trajectory(self, stack_size: int, gamma: float = 0.99, gae_lambda: float = 0.97) -> dict[str, NDArray]:
        cache_key = (stack_size, gamma, gae_lambda)
        if self._cached_flat is not None and self._cache_key == cache_key:
            return self._cached_flat

        done = np.logical_or(self._buffer["terminated"], self._buffer["truncated"])
        returns, advantages = self.compute_returns_and_advantages(
            self._buffer["reward"], self._buffer["critic_value"], done, gamma, gae_lambda,
        )
        
        # normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        obs_windows = self._build_observation_windows(self._buffer["observation"], done, stack_size=stack_size)
        flat = {
            "observation": obs_windows[:, :-1].reshape(-1, stack_size, *self._observation_space),
            "action": self._buffer["action"][:, :-1].reshape(-1, *self._action_space),
            "old_log_probs": self._buffer["old_log_probs"][:, :-1].reshape(-1),
            "old_values": self._buffer["critic_value"][:, :-1].reshape(-1),
            "returns": returns.reshape(-1),
            "advantages": advantages.reshape(-1),
        }
        self._cached_flat = flat
        self._cache_key = cache_key
        return flat

    def get_iid_minibatches(self, minibatch_size: int, stack_size: int, rng: np.random.Generator) -> Iterator[dict[str, NDArray]]:
        flat = self.get_flat_trajectory(stack_size)
        n = flat["returns"].shape[0]
        indices = rng.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = indices[start:start + minibatch_size]
            yield {k: v[idx] for k, v in flat.items()}

    @staticmethod
    def _explained_variance(returns: NDArray[np.float32], values: NDArray[np.float32]) -> float:
        var_returns = np.var(returns)
        if var_returns < 1e-8:
            return float("nan")  # degenerate — targets have no variance to explain
        return float(1 - np.var(returns - values) / var_returns)

    def get_metrics(self, stack_size: int, gamma: float = 0.99, gae_lambda: float = 0.97) -> dict[str, float]:
        flat = self.get_flat_trajectory(stack_size, gamma, gae_lambda)  # reuses cache
        returns = flat["returns"]
        values = self._buffer["critic_value"][:, :-1].reshape(-1)
        actions = flat["action"]
        action_mean = actions.mean(0)
        action_std = actions.std(0)

        metrics = {
            "rollout/value_target_mean": float(returns.mean()),
            "rollout/value_target_std": float(returns.std()),
            "rollout/value_pred_mean": float(values.mean()),
            "rollout/explained_variance": self._explained_variance(returns, values),
        }
        
        metrics.update({
            f"rollout/action_mean_{i}": float(v) for i, v in enumerate(action_mean)
        })

        metrics.update({
            f"rollout/action_std_{i}": float(v) for i, v in enumerate(action_std)
        })

        return metrics

    def reset(self):
        self._buffer = {
            "observation": np.zeros((self._num_envs, self.size) + self._observation_space, self._obs_dtype),
            "action": np.zeros((self._num_envs, self.size) + self._action_space, np.float32),
            "critic_value": np.zeros((self._num_envs, self.size), np.float32),
            "old_log_probs": np.zeros((self._num_envs, self.size), np.float32),
            "reward": np.zeros((self._num_envs, self.size), np.float32),
            "truncated": np.zeros((self._num_envs, self.size), np.bool_),
            "terminated": np.zeros((self._num_envs, self.size), np.bool_),
        }
        self._counter: int = 0
        self._cached_flat = None
        self._cache_key = None
