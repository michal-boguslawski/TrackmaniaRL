# src/rl_lib/envs/reward_wrappers.py
import gymnasium as gym
from typing import Callable


RewardAdjustment = float | Callable[[float], float]


class RewardOnEpisodeEndWrapper(gym.Wrapper):
    """
    Modifies the reward on the step where the episode ends, applying a
    different adjustment depending on whether it ended via termination
    (e.g. crashed / went off-track) or truncation (e.g. time limit hit).

    `on_terminated` / `on_truncated` can each be:
      - None: no adjustment
      - a float: added to the reward
      - a callable: called as adjustment(reward) -> new_reward
    """

    def __init__(
        self,
        env: gym.Env,
        on_terminated: RewardAdjustment | None = None,
        on_truncated: RewardAdjustment | None = None,
        on_win: RewardAdjustment | None = None,
    ):
        super().__init__(env)
        self._on_terminated = on_terminated
        self._on_truncated = on_truncated
        self._on_win = on_win

    @staticmethod
    def _apply(reward: float, adjustment: RewardAdjustment | None) -> float:
        if adjustment is None:
            return reward
        if callable(adjustment):
            return adjustment(reward)
        return reward + adjustment

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated and info.get("lap_finished"):
            reward = self._apply(reward, self._on_win)
        elif terminated:
            reward = self._apply(reward, self._on_terminated)
        elif truncated:
            reward = self._apply(reward, self._on_truncated)

        return observation, reward, terminated, truncated, info
