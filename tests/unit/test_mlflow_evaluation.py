from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rl_lib.evaluation import MLflowModelRef, discover_mlflow_models, evaluate_mlflow, select_mlflow_run
from rl_lib.evaluation.mlflow_evaluation import _shortest_unique_prefixes
from rl_lib.run_config import AgentSettings, RunConfig
from rl_lib.tracking.base import MetricsLogger


OBSERVATION_SHAPE = (96, 96, 1)


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


def _vector_env(num_envs: int = 2, horizon: int = 2):
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import AutoresetMode
    import numpy as np

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

    return gym.vector.SyncVectorEnv(
        [lambda: _EpisodeEnv(horizon) for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


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
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr("rl_lib.evaluation.checkpoint.make_env", lambda config: _vector_env(config.num_envs))
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
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr("rl_lib.evaluation.checkpoint.make_env", lambda config: _vector_env(config.num_envs))

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
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    models = discover_mlflow_models("CarRacing-v3")

    assert [(model.run_id, model.unique_prefix, model.run_name) for model in models] == [
        ("abc111", "a", "old"),
    ]


def test_mlflow_unique_prefix_selection_and_ambiguity(monkeypatch):
    models = [
        MLflowModelRef("abc111", "first", "exp", "abc", 2),
        MLflowModelRef("abd222", "second", "exp", "abd", 1),
    ]
    monkeypatch.setattr("rl_lib.evaluation.mlflow_evaluation.discover_mlflow_models", lambda experiment: models)

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
