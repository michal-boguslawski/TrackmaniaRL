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

from rl_lib.evaluate import (
    MLflowModelRef,
    _shortest_unique_prefixes,
    discover_mlflow_models,
    evaluate_checkpoint,
    evaluate_mlflow,
    select_mlflow_run,
)
from rl_lib.inference import log_evaluation_results, run_inference
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
            "episode": {"r": np.array([101.0, 202.0]), "l": np.array([1, 1])},
        },
        {
            "_episode": np.array([True, True]),
            "final_info": [
                {"episode": {"r": np.array([101.0]), "l": np.array([1])}},
                {"episode": {"r": np.array([202.0]), "l": np.array([1])}},
            ],
        },
    ],
)
def test_run_inference_prefers_pre_transform_episode_rewards_in_info(agent, info):
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

    assert [result.return_ for result in results] == [101.0, 202.0]


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
    monkeypatch.setattr("rl_lib.evaluate.make_env", lambda config: _vector_env(config.num_envs))
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
    monkeypatch.setattr("rl_lib.evaluate.make_env", lambda config: _vector_env(config.num_envs))

    returns = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        episodes=1,
        num_envs=1,
        metrics_loggers=[],
    )

    assert returns == [2.0]


def test_mlflow_evaluation_loads_run_config_model_and_logs(
    tmp_path, monkeypatch, network_config, agent
):
    config = _config(network_config)
    config_path = tmp_path / "run_config.yaml"
    _save_config(config, config_path)
    calls = {}
    fake_mlflow = SimpleNamespace(
        artifacts=SimpleNamespace(
            download_artifacts=lambda **kwargs: str(config_path),
        ),
        pytorch=SimpleNamespace(
            load_model=lambda uri, map_location: calls.update(
                uri=uri, map_location=map_location
            ) or agent.network,
        ),
    )
    monkeypatch.setattr("rl_lib.evaluate._mlflow_module", lambda: fake_mlflow)
    monkeypatch.setattr("rl_lib.evaluate.make_env", lambda config: _vector_env(config.num_envs))
    logger = _Logger()

    returns = evaluate_mlflow(
        run_id="run-abc",
        episodes=2,
        num_envs=2,
        metrics_loggers=[logger],
    )

    assert returns == [2.0, 2.0]
    assert calls["uri"] == "runs:/run-abc/final_model"
    assert logger.metrics[-1][0]["evaluation/mlflow/episodes"] == 2.0


def test_mlflow_run_uri_is_accepted(monkeypatch, tmp_path, network_config, agent):
    config = _config(network_config)
    config_path = tmp_path / "run_config.yaml"
    _save_config(config, config_path)
    calls = []
    fake_mlflow = SimpleNamespace(
        artifacts=SimpleNamespace(download_artifacts=lambda **kwargs: str(config_path)),
        pytorch=SimpleNamespace(
            load_model=lambda uri, map_location: calls.append(uri) or agent.network,
        ),
    )
    monkeypatch.setattr("rl_lib.evaluate._mlflow_module", lambda: fake_mlflow)
    monkeypatch.setattr("rl_lib.evaluate.make_env", lambda config: _vector_env(config.num_envs))

    evaluate_mlflow(
        run_id="runs:/run-xyz/final_model",
        episodes=1,
        num_envs=1,
        metrics_loggers=[],
    )

    assert calls == ["runs:/run-xyz/final_model"]


def test_discovery_lists_only_runs_with_final_model_and_computes_prefixes(monkeypatch):
    runs = [
        SimpleNamespace(
            info=SimpleNamespace(run_id="abc111", start_time=1),
            data=SimpleNamespace(tags={"mlflow.runName": "old"}),
        ),
        SimpleNamespace(
            info=SimpleNamespace(run_id="abd222", start_time=2),
            data=SimpleNamespace(tags={"mlflow.runName": "new"}),
        ),
    ]

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="experiment-id")

        def search_runs(self, experiment_ids):
            return runs

        def list_artifacts(self, run_id):
            if run_id == "abc111":
                return [SimpleNamespace(path="final_model", is_dir=True)]
            return [SimpleNamespace(path="config", is_dir=True)]

    fake_mlflow = SimpleNamespace(tracking=SimpleNamespace(MlflowClient=Client))
    monkeypatch.setattr("rl_lib.evaluate._mlflow_module", lambda: fake_mlflow)

    models = discover_mlflow_models("CarRacing-v3")

    assert [(model.run_id, model.unique_prefix, model.run_name) for model in models] == [
        ("abc111", "a", "old"),
    ]


def test_mlflow_unique_prefix_selection_and_ambiguity(monkeypatch):
    models = [
        MLflowModelRef("abc111", "first", "exp", "abc", 2),
        MLflowModelRef("abd222", "second", "exp", "abd", 1),
    ]
    monkeypatch.setattr("rl_lib.evaluate.discover_mlflow_models", lambda experiment: models)

    selected = select_mlflow_run("exp", identifier="abc")
    assert selected.run_name == "first"
    assert select_mlflow_run("exp", input_fn=lambda prompt: "abd").run_id == "abd222"
    with pytest.raises(ValueError, match="ambiguous"):
        select_mlflow_run("exp", identifier="a")


def test_shortest_unique_prefixes_handle_duplicate_ids():
    assert _shortest_unique_prefixes(["aaaa", "aaab", "bbbb"]) == {
        "aaaa": "aaaa",
        "aaab": "aaab",
        "bbbb": "b",
    }
