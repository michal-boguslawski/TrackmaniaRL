"""Reward shaping wrapper for episode-end rewards.

Provides RewardOnEpisodeEndWrapper to apply custom reward adjustments
when an episode terminates, truncates, or wins (completes a lap).
"""

# src/rl_lib/envs/reward_wrappers.py
import gymnasium as gym
from typing import Callable


RewardAdjustment = float | Callable[[float], float]


class RewardOnEpisodeEndWrapper(gym.Wrapper):
    """Modify reward on the terminal step based on termination type.

    Applies different reward adjustments depending on how the episode ended:
    - Termination (crash/off-track): on_terminated adjustment
    - Truncation (time limit): on_truncated adjustment
    - Win (lap finished): on_win adjustment

    Each adjustment can be:
      - None: no adjustment
      - float: added to the reward
      - callable: called as adjustment(reward) -> new_reward

    Attributes:
        _on_terminated: Adjustment for terminated episodes.
        _on_truncated: Adjustment for truncated episodes.
        _on_win: Adjustment for winning episodes (lap_finished in info).
    """

    def __init__(
        self,
        env: gym.Env,
        on_terminated: RewardAdjustment | None = None,
        on_truncated: RewardAdjustment | None = None,
        on_win: RewardAdjustment | None = None,
    ):
        """Initialize reward wrapper.

        Args:
            env: Base environment.
            on_terminated: Reward adjustment for termination.
            on_truncated: Reward adjustment for truncation.
            on_win: Reward adjustment for win (lap finished).
        """
        super().__init__(env)
        self._on_terminated = on_terminated
        self._on_truncated = on_truncated
        self._on_win = on_win

    @staticmethod
    def _apply(reward: float, adjustment: RewardAdjustment | None) -> float:
        """Apply adjustment to reward."""
        if adjustment is None:
            return reward
        if callable(adjustment):
            return adjustment(reward)
        return reward + adjustment

    def step(self, action):
        """Step environment and apply reward adjustment on episode end."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated and info.get("lap_finished"):
            reward = self._apply(reward, self._on_win)
        elif terminated:
            reward = self._apply(reward, self._on_terminated)
        elif truncated:
            reward = self._apply(reward, self._on_truncated)

        return observation, reward, terminated, truncated, info
