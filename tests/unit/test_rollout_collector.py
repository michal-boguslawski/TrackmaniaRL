"""Unit tests for the rollout collector.

Uses a tiny custom vector env so the tests stay fast and deterministic while
still exercising the real buffer/trainer/callback wiring, including the
truncation bootstrap path.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as T
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.training.callbacks.base import Callback
from rl_lib.training.rollout_collector import RolloutCollector
from rl_lib.run_config import RolloutSettings, TrainerSettings
from rl_lib.training.ppo.trainer import PPOTrainer


OBSERVATION_SHAPE = (96, 96, 1)
NUM_ENVS = 2
ROLLOUT_SIZE = 4


class _CountingEnv(gym.Env):
    """Emits an increasing counter and terminates/truncates on a schedule."""

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 3, terminate: bool = False):
        self.observation_space = spaces.Box(0, 255, OBSERVATION_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._horizon = horizon
        self._terminate = terminate
        self._t = 0
        self._episode_reward = 0.0

    def _observation(self) -> np.ndarray:
        value = (self._t % 255)
        return np.full(OBSERVATION_SHAPE, value, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._episode_reward = 0.0
        return self._observation(), {"tick": 0}

    def step(self, action):
        self._t += 1
        reward = float(self._t)
        self._episode_reward += reward
        ended = self._t >= self._horizon
        truncated = ended and not self._terminate
        terminated = ended and self._terminate
        info = {
            "tick": self._t,
            "reward_so_far": self._episode_reward,
            "final_obs": self._observation() if ended else None,
        }
        return self._observation(), reward, terminated, truncated, info


class _SpyCallback(Callback):
    def __init__(self):
        self.events: list[tuple] = []

    def on_rollout_start(self, *args, **kwargs):
        self.events.append(("rollout_start", kwargs.get("config")))

    def on_env_step(self, *args, **kwargs):
        info = kwargs.get("info", {})
        # on an autoreset step the terminal payload is nested under final_info
        ticks = info.get("final_info", info).get("tick")
        self.events.append(
            ("env_step", kwargs.get("step"), np.asarray(ticks).tolist())
        )

    def on_rollout_end(self, *args, **kwargs):
        self.events.append(("rollout_end",))

    def flush(self, *args, **kwargs):
        self.events.append(("flush",))


@pytest.fixture
def vector_env():
    """Factory for a SAME_STEP vector env; every env is closed at teardown."""
    created: list[gym.vector.SyncVectorEnv] = []

    def make(horizon: int = 3, terminate: bool = False) -> gym.vector.SyncVectorEnv:
        def factory():
            return _CountingEnv(horizon=horizon, terminate=terminate)

        env = gym.vector.SyncVectorEnv(
            [factory for _ in range(NUM_ENVS)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        created.append(env)
        return env

    yield make

    for env in created:
        env.close()


def _value(collector: RolloutCollector, key: str, step: int) -> T.Tensor:
    """Unpadded per-step field, indexed by the step that produced it."""
    return collector.buffer._ordered(key)[:, step]


def _flag(collector: RolloutCollector, key: str, step: int) -> T.Tensor:
    """Padded per-step flag, indexed by the step that produced it."""
    return collector.buffer._ordered(key, windowed=True)[
        :, step + collector.trainer.stack_size - 1
    ]


def _collector(vector_env, agent, buffer_size=3, minibatch_size=2, **env_kwargs) -> RolloutCollector:
    """A collector whose buffer and collector share one RolloutSettings, so
    epochs/minibatch_size and gamma/gae_lambda cannot drift apart."""
    settings = RolloutSettings(
        buffer_size=buffer_size, epochs=1, minibatch_size=minibatch_size
    )
    return RolloutCollector(
        env=vector_env(**env_kwargs),
        buffer=RolloutBuffer(settings, stack_size=agent.stack_size),
        trainer=PPOTrainer(agent, TrainerSettings()),
        config=settings,
    )


@pytest.fixture
def collector(vector_env, agent) -> RolloutCollector:
    return _collector(
        vector_env, agent, buffer_size=ROLLOUT_SIZE, minibatch_size=4
    )


def test_run_collects_a_full_rollout_and_trains(collector: RolloutCollector, agent):
    collector.run(training_steps=ROLLOUT_SIZE)

    assert collector.buffer._counter == 0, "the counter is reset after training"
    assert collector.trainer._step == (
        1 + (ROLLOUT_SIZE - 1) * NUM_ENVS // 4
    ) * 1, "one minibatch epoch of updates"


def test_run_keeps_collecting_until_the_buffer_fills(collector: RolloutCollector):
    collector.run(training_steps=ROLLOUT_SIZE + 1)

    # the extra step stays in the buffer for the next rollout
    assert collector.buffer._counter == 1
    assert collector.buffer.is_full() is False


def test_run_advances_optional_profiler_once_per_vector_step(collector: RolloutCollector):
    class ProfilerSpy:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    profiler = ProfilerSpy()
    collector.run(training_steps=3, profiler=profiler)

    assert profiler.steps == 3


def test_run_resets_the_environment_and_buffer(collector: RolloutCollector):
    collector.buffer.add  # sanity: the buffer API exists
    from rl_lib.buffers.rollout_buffer import RolloutStep

    collector.buffer.add(
        RolloutStep(
            observation=T.zeros(NUM_ENVS, 96, 96, 1, dtype=T.uint8),
            action=T.zeros(NUM_ENVS, 3),
            critic_value=T.zeros(NUM_ENVS),
            old_log_probs=T.zeros(NUM_ENVS, 3),
            reward=T.zeros(NUM_ENVS),
            terminated=T.zeros(NUM_ENVS, dtype=T.bool),
            truncated=T.zeros(NUM_ENVS, dtype=T.bool),
            truncated_value=T.zeros(NUM_ENVS),
        )
    )
    assert collector.buffer._counter == 1

    collector.run(training_steps=1)

    assert collector.buffer._counter == 1, "the stale step was dropped at startup"


def test_run_drives_every_callback_hook(collector: RolloutCollector):
    callback = _SpyCallback()
    collector._callbacks = type(collector._callbacks)([callback])

    collector.run(training_steps=ROLLOUT_SIZE)

    kinds = [event[0] for event in callback.events]
    assert kinds[0] == "rollout_start"
    assert kinds[-1] == "rollout_end"
    assert kinds.count("env_step") == ROLLOUT_SIZE
    assert kinds.count("flush") == 1, "a flush per completed rollout"

    config = callback.events[0][1]
    assert config["training_steps"] == ROLLOUT_SIZE
    assert config["num_envs"] == NUM_ENVS
    assert config["epochs"] == collector.cfg.epochs
    assert config["minibatch_size"] == collector.cfg.minibatch_size
    assert any(key.startswith("buffer.") for key in config)
    assert any(key.startswith("trainer.") for key in config)
    assert any(key.startswith("network.") for key in config)


def test_env_step_callbacks_see_the_environment_step_counter(collector: RolloutCollector):
    callback = _SpyCallback()
    collector._callbacks = type(collector._callbacks)([callback])

    collector.run(training_steps=ROLLOUT_SIZE)

    # the env's horizon is 3 and it autoresets, so the counters cycle 1, 2, 3
    seen = [event[2] for event in callback.events if event[0] == "env_step"]
    assert seen == [[1, 1], [2, 2], [3, 3], [1, 1]]
    assert [event[1] for event in callback.events if event[0] == "env_step"] == [0, 1, 2, 3]


def test_truncation_bootstraps_from_the_final_observation(vector_env, agent):
    collector = _collector(vector_env, agent, horizon=2)

    collector.run(training_steps=3)

    # observation/terminated/truncated carry `stack_size - 1` leading pad steps,
    # the remaining fields are unpadded, so index k of a value field is the
    # padded index k + stack_size - 1 of the flag fields
    assert _flag(collector, "truncated", 0).tolist() == [False, False]
    assert _flag(collector, "truncated", 1).tolist() == [True, True]
    assert _value(collector, "truncated_value", 1).abs().sum() > 0.0, (
        "a truncated step bootstraps from the final observation"
    )
    # the fresh episode after the autoreset is not truncated
    assert _flag(collector, "truncated", 2).tolist() == [False, False]
    assert _value(collector, "truncated_value", 2).tolist() == [0.0, 0.0]


def test_termination_does_not_bootstrap(vector_env, agent):
    collector = _collector(vector_env, agent, horizon=2, terminate=True)

    collector.run(training_steps=3)

    assert _flag(collector, "terminated", 1).tolist() == [True, True]
    assert _flag(collector, "truncated", 1).tolist() == [False, False]
    assert _value(collector, "truncated_value", 1).tolist() == [0.0, 0.0]


def test_update_rebuilds_the_same_temporal_masks_as_collection(
    vector_env, agent, monkeypatch
):
    """PPO's importance ratio is only meaningful if the update re-encodes the
    history the action was actually taken with, so the mask the buffer hands to
    the update has to match the one `Agent.act` used, episode boundaries
    included. A buffer large enough to hold the whole run keeps the weights
    fixed, so the two paths are directly comparable."""

    horizon = 3
    steps = 2 * horizon + 1
    collector = _collector(vector_env, agent, buffer_size=steps + 1, horizon=horizon)

    masks: list[T.Tensor] = []
    original = agent._get_mask_window

    def spy(done: T.Tensor) -> T.Tensor:
        mask = original(done)
        masks.append(mask.clone())
        return mask

    monkeypatch.setattr(agent, "_get_mask_window", spy)
    collector.run(training_steps=steps)

    batch = collector.buffer.get()
    stack_size = agent.stack_size

    for transition, collected in enumerate(masks[:-1]):
        for env in range(NUM_ENVS):
            window = batch["dones"][env, transition : transition + stack_size]
            rebuilt = window.logical_not() & (window.flip(0).cumsum(0).flip(0) > 0)
            T.testing.assert_close(
                rebuilt, collected[env], atol=0, rtol=0
            ), f"env {env}, transition {transition}"

    # the horizon is crossed inside this rollout, so the comparison is not vacuous
    assert any(mask.any() for mask in masks), "no frame was ever masked out"
    assert batch["dones"].any(), "the rollout contains no episode boundary"


def test_missing_final_observation_raises(vector_env, agent, monkeypatch):
    collector = _collector(vector_env, agent, horizon=2)
    monkeypatch.setattr(
        type(collector.env),
        "step",
        lambda self, action: (
            np.zeros((NUM_ENVS, *OBSERVATION_SHAPE), np.uint8),
            np.ones(NUM_ENVS, np.float32),
            np.zeros(NUM_ENVS, bool),
            np.ones(NUM_ENVS, bool),
            {},
        ),
    )

    with pytest.raises(RuntimeError, match="final observation"):
        collector.run(training_steps=1)


def test_incomplete_final_observation_mask_raises(vector_env, agent, monkeypatch):
    collector = _collector(vector_env, agent, horizon=2)
    original_step = type(collector.env).step

    def _step(self, action):
        observation, reward, terminated, truncated, info = original_step(self, action)
        info["_final_obs"] = np.array([False] * NUM_ENVS)
        return observation, reward, terminated, truncated, info

    monkeypatch.setattr(type(collector.env), "step", _step)

    with pytest.raises(RuntimeError, match="every truncated episode"):
        collector.run(training_steps=2)


def test_config_merges_the_buffer_trainer_and_network_settings(collector: RolloutCollector):
    config = collector.config()

    assert config["epochs"] == 1
    assert config["minibatch_size"] == 4
    assert config["num_envs"] == NUM_ENVS
    assert "training_steps" not in config
    assert config["buffer.size"] == collector.buffer.size
    assert config["buffer.stack_size"] == collector.trainer.stack_size
    assert config["trainer.ppo_epsilon"] == collector.trainer.cfg.ppo_epsilon
    assert all(
        isinstance(value, (int, float, str, bool)) for value in config.values()
    ), "the config has to stay flat and loggable"


def test_repr_mentions_the_run_shape(collector: RolloutCollector):
    text = repr(collector)

    assert "RolloutCollector(" in text
    assert "epochs=1" in text
    assert "minibatch_size=4" in text
    assert "num_envs=2" in text
