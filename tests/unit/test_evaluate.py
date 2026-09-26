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

from rl_lib.evaluation import evaluate_checkpoint
from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.run_config import (
    AgentSettings,
    EnvironmentSettings,
    EvaluationCallbackSettings,
    RunConfig,
)
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.evaluate import EvaluationCallback


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


def test_evaluation_callback_runs_on_vector_step_intervals_and_at_end(
    monkeypatch, agent
):
    env = _vector_env(num_envs=2, horizon=1)
    monkeypatch.setattr("rl_lib.training.callbacks.evaluate.make_env", lambda config: env)
    logger = _Logger()
    callback = EvaluationCallback(
        agent,
        [logger],
        EvaluationCallbackSettings(
            environment=EnvironmentSettings(id="unused", num_envs=2),
            interval=2,
            episodes=1,
            final_episodes=3,
            seed=7,
        ),
    )
    callback.on_rollout_start(config={"training_steps": 5})

    for step in range(5):
        callback.on_env_step(step=step)
    callback.on_rollout_end()

    summaries = [
        (metrics, step)
        for metrics, step in logger.metrics
        if "evaluation/periodic/episodes" in metrics
        or "evaluation/final/episodes" in metrics
    ]
    assert [
        (
            step,
            metrics[
                "evaluation/periodic/episodes"
                if "evaluation/periodic/episodes" in metrics
                else "evaluation/final/episodes"
            ],
        )
        for metrics, step in summaries
    ] == [
        (2, 1.0),
        (4, 1.0),
        (5, 3.0),
    ]
    assert env.closed


def test_local_checkpoint_evaluation_uses_shared_inference(
    tmp_path, monkeypatch, network_config, agent
):
    config = _config(network_config)
    config_path = tmp_path / "run.yaml"
    _save_config(config, config_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    agent.save_state_dict(str(checkpoint_path))
    monkeypatch.setattr("rl_lib.evaluation.checkpoint.make_env", lambda config: _vector_env(config.num_envs))
    logger = _Logger()

    returns = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        episodes=3,
        num_envs=2,
        metrics_loggers=[logger],
    )

    assert returns == [2.0, 2.0, 2.0]
    assert logger.metrics[-1][0]["evaluation/local/episodes"] == 3.0


def test_local_evaluation_accepts_the_full_model_saved_at_training_end(
    tmp_path, monkeypatch, network_config, agent
):
    config_path = tmp_path / "run.yaml"
    _save_config(_config(network_config), config_path)
    checkpoint_path = tmp_path / "final_model.pt"
    T.save(agent.network, checkpoint_path)
    monkeypatch.setattr("rl_lib.evaluation.checkpoint.make_env", lambda config: _vector_env(config.num_envs))

    returns = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        episodes=1,
        num_envs=1,
        metrics_loggers=[],
    )

    assert returns == [2.0]