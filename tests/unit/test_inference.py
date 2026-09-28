from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch as T
import yaml
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.run_config import (
    AgentSettings,
    EnvironmentSettings,
    EvaluationCallbackSettings,
    RunConfig,
)
from rl_lib.tracking.base import MetricsLogger


OBSERVATION_SHAPE = (96, 96, 1)


class _EpisodeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 2):
        self.observation_space = spaces.Box(0, 255, OBSERVATION_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.horizon = horizon
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(OBSERVATION_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        self.step_count += 1
        ended = self.step_count == self.horizon
        observation = np.full(OBSERVATION_SHAPE, self.step_count, dtype=np.uint8)
        return observation, 1.0, ended, False, {}


def _vector_env(num_envs: int = 2, horizon: int = 2):
    return gym.vector.SyncVectorEnv(
        [lambda: _EpisodeEnv(horizon) for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


class _Logger(MetricsLogger):
    def __init__(self):
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


def _config(network_config, num_envs: int = 2) -> RunConfig:
    return RunConfig.model_validate({
        "environment": {
            "id": "test-episode-env",
            "num_envs": num_envs,
            "normalize_rewards": True,
            "wrappers": [],
        },
        "agent": AgentSettings(stack_size=2).model_dump(),
        "network": network_config.model_dump(),
        "callbacks": {
            "episode_statistics": False,
            "record_video": False,
            "evaluation": False,
        },
    })


def _save_config(config: RunConfig, path: Path) -> None:
    raw = config.model_dump(mode="json", exclude_unset=True, exclude={"runtime"})
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_run_inference_collects_exact_episode_budget_across_envs(agent):
    env = _vector_env(num_envs=2, horizon=2)
    agent.train()
    try:
        results = run_inference(agent, env, episodes=3, seed=5, temperature=0.0)
    finally:
        env.close()

    assert [(result.return_, result.length) for result in results] == [(2.0, 2)] * 3
    assert agent.network.training is True
    assert not agent._obs_window
    assert not agent._done_window


def test_run_inference_preserves_eval_mode(agent):
    env = _vector_env(num_envs=1, horizon=1)
    agent.eval()
    try:
        run_inference(agent, env, episodes=1)
    finally:
        env.close()

    assert agent.network.training is False


@pytest.mark.parametrize(
    "info",
    [
        {
            "_episode": np.array([True, True]),
            "episode": {"r": np.array([101.0, 202.0]), "l": np.array([11, 22])},
        },
        {
            "_episode": np.array([True, True]),
            "final_info": [
                {"episode": {"r": np.array([101.0]), "l": np.array([11])}},
                {"episode": {"r": np.array([202.0]), "l": np.array([22])}},
            ],
        },
    ],
)
def test_run_inference_prefers_episode_statistics_in_info(agent, info):
    class InfoRewardEnv:
        num_envs = 2

        def reset(self, seed=None):
            return np.zeros((2, *OBSERVATION_SHAPE), dtype=np.uint8), {}

        def step(self, action):
            return (
                np.zeros((2, *OBSERVATION_SHAPE), dtype=np.uint8),
                np.array([-5.0, -7.0], dtype=np.float32),
                np.ones(2, dtype=np.bool_),
                np.zeros(2, dtype=np.bool_),
                info,
            )

    results = run_inference(agent, InfoRewardEnv(), episodes=2)

    assert [(result.return_, result.length) for result in results] == [
        (101.0, 11),
        (202.0, 22),
    ]


def test_log_evaluation_results_sends_each_episode_and_summary_to_all_loggers():
    first, second = _Logger(), _Logger()

    summary = log_evaluation_results(
        [
            SimpleNamespace(return_=2.0, length=4),
            SimpleNamespace(return_=6.0, length=8),
        ],
        [first, second],
        step=12,
        scope="periodic",
    )

    assert len(first.metrics) == len(second.metrics) == 3
    assert first.metrics[-1] == (
        {
            "evaluation/periodic/return_mean": 4.0,
            "evaluation/periodic/return_std": 2.0,
            "evaluation/periodic/return_min": 2.0,
            "evaluation/periodic/return_max": 6.0,
            "evaluation/periodic/length_mean": 6.0,
            "evaluation/periodic/episodes": 2.0,
        },
        12,
    )
    assert summary["evaluation/periodic/return_mean"] == 4.0
