"""Unit tests for the environment factory, the wrapper registry and the
custom reward wrapper.

`make_vec` is stubbed out so the tests assert on the plumbing (wrapper
construction, kwargs, autoreset mode) without booting a game.
"""

from __future__ import annotations

import importlib

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium import VectorizeMode
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)
from gymnasium.wrappers.vector import NormalizeReward

from rl_lib.envs.wrappers.registry import WRAPPERS
from rl_lib.envs.wrappers.reward_wrappers import RewardOnEpisodeEndWrapper


make_env_module = importlib.import_module("rl_lib.envs.make_env")
make_env = make_env_module.make_env


class _PixelEnv(gym.Env):
    """RGB uint8 observations, so the image wrappers have something to chew on."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        horizon: int = 4,
        lap_on_last: bool = False,
        render_mode=None,
        terminate: bool = False,
        continuous: bool = True,
    ):
        self.continuous = continuous
        self.observation_space = spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
        self.action_space = (
            spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            if continuous
            else spaces.Discrete(3)
        )
        self._horizon = horizon
        self._lap_on_last = lap_on_last
        self._terminate = terminate
        self._t = 0
        self.render_mode = render_mode

    def _observation(self) -> np.ndarray:
        return np.full((96, 96, 3), self._t % 256, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._observation(), {}

    def step(self, action):
        self._t += 1
        ended = self._t >= self._horizon
        info = {"lap_finished": ended and self._lap_on_last}
        return self._observation(), 1.0, ended and self._terminate, ended and not self._terminate, info


ENV_ID = "RewardWrapperFactoryTest-v0"
if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=_PixelEnv, max_episode_steps=3)


@pytest.fixture
def capture_make_vec(monkeypatch):
    """Replace `make_vec` with a stub and expose the captured keyword arguments."""

    captured: dict = {}

    def fake_make_vec(env_id, **kwargs):
        captured["env_id"] = env_id
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(make_env_module, "make_vec", fake_make_vec)
    return captured


# ---------------------------------------------------------------- make_env


def test_make_env_passes_the_requested_shape(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=4, vectorization_mode="sync")

    assert capture_make_vec["env_id"] == "SomeEnv-v0"
    assert capture_make_vec["num_envs"] == 4
    assert capture_make_vec["vectorization_mode"] is VectorizeMode.SYNC
    assert capture_make_vec["vector_kwargs"]["autoreset_mode"] is AutoresetMode.SAME_STEP
    assert capture_make_vec["continuous"] is True
    assert capture_make_vec["render_mode"] is None
    assert capture_make_vec["wrappers"] == []


def test_make_env_defaults_to_async_without_render_mode(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=1)

    assert capture_make_vec["vectorization_mode"] is VectorizeMode.ASYNC


def test_make_env_recording_forces_sync_and_rgb(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=1, record=True, vectorization_mode="async")

    assert capture_make_vec["vectorization_mode"] is VectorizeMode.SYNC
    assert capture_make_vec["render_mode"] == "rgb_array"
    assert [wrapper.func for wrapper in capture_make_vec["wrappers"]] == [WRAPPERS["record_video"]]


def test_make_env_inserts_the_video_wrapper_first(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=1, record=True, wrappers=["grayscale"])

    assert [wrapper.func for wrapper in capture_make_vec["wrappers"]] == [
        WRAPPERS["record_video"],
        WRAPPERS["grayscale"],
    ]


def test_make_env_does_not_mutate_the_callers_wrapper_list(capture_make_vec):
    wrappers = ["grayscale"]

    make_env("SomeEnv-v0", num_envs=1, record=True, wrappers=wrappers)

    assert wrappers == ["grayscale"]


def test_make_env_forwards_the_video_folder_and_skip(capture_make_vec):
    make_env(
        "SomeEnv-v0",
        num_envs=1,
        record=True,
        skip=7,
        video_folder="tmp/videos",
    )

    wrapper = capture_make_vec["wrappers"][0]
    assert wrapper.keywords["video_folder"] == "tmp/videos"
    assert wrapper.keywords["skip"] == 7
    assert wrapper.keywords["video_length"] == 0
    assert wrapper.keywords["video_name_prefix"] == "rl-video"


def test_make_env_forwards_typed_wrapper_specific_options(capture_make_vec):
    make_env(
        "SomeEnv-v0",
        num_envs=1,
        wrappers=[
            {"name": "record_episode_stats", "stats_buffer_length": 25, "stats_key": "episode_stats"},
            {"name": "max_and_skip", "skip": 3},
        ],
    )

    stats_wrapper, skip_wrapper = capture_make_vec["wrappers"]
    assert stats_wrapper.keywords["stats_buffer_length"] == 25
    assert stats_wrapper.keywords["stats_key"] == "episode_stats"
    assert skip_wrapper.keywords["skip"] == 3


def test_make_env_wraps_with_normalized_rewards():
    env = make_env(ENV_ID, num_envs=2, vectorization_mode="sync", normalize_rewards=True)

    try:
        assert isinstance(env, NormalizeReward)
        env.reset()
        _, reward, *_ = env.step(np.zeros((2, 3), dtype=np.float32))
        assert reward.shape == (2,)
    finally:
        env.close()


def test_make_env_leaves_rewards_untouched_by_default(capture_make_vec, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(make_env_module, "make_vec", lambda env_id, **kwargs: sentinel)

    assert make_env("SomeEnv-v0", num_envs=2) is sentinel


def test_make_env_disables_continuous_for_discrete_envs(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=1, continuous=False)

    assert capture_make_vec["continuous"] is False


# ---------------------------------------------------------------- registry


def test_registry_exposes_the_documented_wrappers():
    assert set(WRAPPERS) == {
        "grayscale",
        "frame_stack",
        "record_episode_stats",
        "record_video",
        "max_and_skip",
        "reward_on_done",
    }


def test_grayscale_wrapper_keeps_the_channel_dimension():
    env = WRAPPERS["grayscale"](_PixelEnv(), video_folder="v", skip=4)

    assert isinstance(env, GrayscaleObservation)
    env.reset()
    observation, *_ = env.step(env.action_space.sample())
    assert observation.shape == (96, 96, 1)


def test_frame_stack_wrapper_stacks_observations():
    env = WRAPPERS["frame_stack"](_PixelEnv(), stack_size=3, video_folder="v", skip=4)

    assert isinstance(env, FrameStackObservation)
    env.reset()
    observation, *_ = env.step(env.action_space.sample())
    assert observation.shape == (3, 96, 96, 3)


def test_record_episode_statistics_wrapper_reports_episode_returns():
    env = WRAPPERS["record_episode_stats"](_PixelEnv(horizon=2), video_folder="v", skip=4)

    assert isinstance(env, RecordEpisodeStatistics)
    env.reset()
    env.step(env.action_space.sample())
    _, _, terminated, truncated, info = env.step(env.action_space.sample())

    assert terminated or truncated
    assert info["episode"]["r"] == pytest.approx(2.0)
    assert info["episode"]["l"] == 2


def test_max_and_skip_wrapper_accumulates_the_reward():
    env = WRAPPERS["max_and_skip"](_PixelEnv(horizon=8), skip=3, video_folder="v")

    assert isinstance(env, MaxAndSkipObservation)
    env.reset()
    _, reward, *_ = env.step(env.action_space.sample())
    assert reward == pytest.approx(3.0)


def test_record_video_wrapper_uses_the_given_folder(tmp_path):
    env = WRAPPERS["record_video"](
        _PixelEnv(render_mode="rgb_array"),
        video_folder=str(tmp_path),
        skip=1,
    )

    assert isinstance(env, RecordVideo)
    assert str(tmp_path) in env.video_folder


def test_reward_on_done_registry_entry_applies_the_truncation_penalty():
    env = WRAPPERS["reward_on_done"](
        _PixelEnv(horizon=1), on_truncated=-10, video_folder="v", skip=4
    )

    assert isinstance(env, RewardOnEpisodeEndWrapper)
    env.reset()
    _, reward, _, truncated, _ = env.step(env.action_space.sample())

    assert truncated
    assert reward == pytest.approx(-9.0)


# ---------------------------------------------------------------- reward wrapper


def _wrapped_env(**kwargs) -> RewardOnEpisodeEndWrapper:
    return RewardOnEpisodeEndWrapper(
        _PixelEnv(horizon=1, lap_on_last=True, terminate=True), **kwargs
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 1.0),
        ({"on_win": 100.0}, 101.0),
        ({"on_win": -50.0}, -49.0),
        ({"on_win": lambda reward: reward * 3}, 3.0),
        ({"on_win": None}, 1.0),
    ],
)
def test_lap_finish_uses_the_win_adjustment(kwargs, expected):
    env = _wrapped_env(**kwargs)
    env.reset()

    _, reward, terminated, _, info = env.step(env.action_space.sample())

    assert terminated and info["lap_finished"]
    assert reward == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"on_terminated": -1.0}, 0.0),
        ({"on_terminated": 5.0}, 6.0),
        ({"on_terminated": lambda reward: -reward}, -1.0),
        ({"on_terminated": None}, 1.0),
    ],
)
def test_termination_without_a_lap_uses_the_termination_adjustment(kwargs, expected):
    env = RewardOnEpisodeEndWrapper(_PixelEnv(horizon=1, terminate=True), **kwargs)
    env.reset()

    _, reward, terminated, _, info = env.step(env.action_space.sample())

    assert terminated and not info["lap_finished"]
    assert reward == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"on_truncated": -10.0}, -9.0),
        ({"on_truncated": 2.0}, 3.0),
        ({"on_truncated": lambda reward: abs(reward)}, 1.0),
        ({"on_truncated": None}, 1.0),
    ],
)
def test_truncation_uses_the_truncation_adjustment(kwargs, expected):
    env = RewardOnEpisodeEndWrapper(_PixelEnv(horizon=1), **kwargs)
    env.reset()

    _, reward, _, truncated, _ = env.step(env.action_space.sample())

    assert truncated
    assert reward == pytest.approx(expected)


def test_only_the_final_step_is_adjusted():
    env = RewardOnEpisodeEndWrapper(
        _PixelEnv(horizon=3), on_terminated=-100.0, on_truncated=-100.0
    )
    env.reset()

    rewards = [env.step(env.action_space.sample())[1] for _ in range(3)]

    assert rewards == [1.0, 1.0, pytest.approx(-99.0)]


def test_win_adjustment_wins_over_termination():
    env = RewardOnEpisodeEndWrapper(
        _PixelEnv(horizon=1, lap_on_last=True, terminate=True), on_terminated=-7.0
    )
    env.reset()

    _, reward, terminated, _, _ = env.step(env.action_space.sample())

    assert terminated
    assert reward == pytest.approx(1.0)


def test_the_wrapper_passes_everything_else_through():
    env = RewardOnEpisodeEndWrapper(_PixelEnv(horizon=4), on_win=5.0)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert observation.shape == (96, 96, 3)
    assert (reward, terminated, truncated) == (1.0, False, False)
    assert info == {"lap_finished": False}


def test_apply_is_a_pure_function():
    assert RewardOnEpisodeEndWrapper._apply(2.0, None) == 2.0
    assert RewardOnEpisodeEndWrapper._apply(2.0, 3.0) == 5.0
    assert RewardOnEpisodeEndWrapper._apply(2.0, -3.0) == -1.0
    assert RewardOnEpisodeEndWrapper._apply(2.0, lambda value: value / 2) == 1.0


def test_frame_stack_is_usable_through_make_env(capture_make_vec):
    make_env("SomeEnv-v0", num_envs=1, wrappers=[{"name": "frame_stack", "stack_size": 2}])

    capture_make_vec["wrappers"][0](_PixelEnv())
