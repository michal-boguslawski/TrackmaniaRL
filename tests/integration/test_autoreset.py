"""Integration coverage for Gymnasium vector autoreset and CarRacing."""

from importlib import import_module

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

from rl_lib.run_config import EnvironmentSettings


class OneStepEnv(gym.Env):
    def __init__(self, continuous=True, render_mode=None):
        self.observation_space = spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        return np.array([1], dtype=np.uint8), 1.0, False, False, {}


def test_same_step_autoreset_does_not_emit_reset_only_steps():
    test_env_id = "MakeEnvAutoresetIntegration-v0"
    if test_env_id not in gym.registry:
        gym.register(test_env_id, entry_point=OneStepEnv, max_episode_steps=1)

    make_env = import_module("rl_lib.envs.make_env").make_env
    env = make_env(EnvironmentSettings(
        id=test_env_id, num_envs=1, vectorization_mode="sync", normalize_rewards=False, wrappers=[]
    ))

    try:
        assert env.autoreset_mode is AutoresetMode.SAME_STEP
        observation, _ = env.reset()

        # Every call executes an action and yields its reward, including the
        # call immediately after a truncation; no reset-only step is emitted.
        for _ in range(2):
            observation, reward, terminated, truncated, info = env.step(np.array([0]))
            np.testing.assert_array_equal(observation, np.array([[0]], dtype=np.uint8))
            np.testing.assert_array_equal(reward, np.array([1.0]))
            np.testing.assert_array_equal(terminated, np.array([False]))
            np.testing.assert_array_equal(truncated, np.array([True]))
            final_observation = info.get("final_obs", info.get("final_observation"))
            assert final_observation is not None
            np.testing.assert_array_equal(final_observation[0], np.array([1], dtype=np.uint8))
    finally:
        env.close()


def test_car_racing_vector_env_resets_and_steps():
    make_env = import_module("rl_lib.envs.make_env").make_env
    env = make_env(EnvironmentSettings(
        id="CarRacing-v3", num_envs=1, vectorization_mode="sync", skip=1, normalize_rewards=False, wrappers=[]
    ))

    try:
        assert env.autoreset_mode is AutoresetMode.SAME_STEP
        observation, _ = env.reset(seed=7)
        assert observation.shape == (1, 96, 96, 3)
        assert observation.dtype == np.uint8

        next_observation, reward, terminated, truncated, _ = env.step(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        )
        assert next_observation.shape == observation.shape
        assert np.isfinite(reward).all()
        assert terminated.shape == truncated.shape == (1,)
    finally:
        env.close()
