import json

import pytest
import torch as T
import yaml

from rl_lib.tracking.local_artifact_logger import LocalArtifactLogger


def test_local_logger_persists_state_dict_and_model_under_artifact_paths(tmp_path):
    logger = LocalArtifactLogger(tmp_path / "run")
    model = T.nn.Linear(2, 1)
    state_dict = model.state_dict()

    logger.log_state_dict(state_dict, artifact_path="checkpoints/step_20")
    logger.log_state_dict(state_dict, artifact_path="final_state_dict")
    logger.log_model(model, artifact_path="final_model")

    checkpoint = T.load(
        tmp_path / "run" / "checkpoints" / "step_020.pt",
        weights_only=True,
    )
    final_state_dict = T.load(
        tmp_path / "run" / "final_state_dict.pt",
        weights_only=True,
    )
    final_model = T.load(
        tmp_path / "run" / "final_model.pt",
        weights_only=False,
    )

    for key, tensor in state_dict.items():
        T.testing.assert_close(checkpoint[key], tensor)
        T.testing.assert_close(final_state_dict[key], tensor)
    assert isinstance(final_model, T.nn.Linear)


def test_local_logger_persists_parameters_config_and_final_evaluation(tmp_path):
    logger = LocalArtifactLogger(tmp_path / "run")
    parameters = {"training_steps": 100, "seed": 7}
    config = {"run": {"seed": 7}}
    episodes = [{"return": 3.5, "length": 12}, {"return": 4.5, "length": 14}]
    summary = {"evaluation/final/return_mean": 4.0}

    logger.log_parameters(parameters)
    logger.log_config(config)
    logger.log_evaluation(episodes, summary, step=100, scope="final")

    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "config/parameters.json").read_text()) == parameters
    assert yaml.safe_load((run_dir / "config/run_config.yaml").read_text()) == config
    saved_evaluation = json.loads(
        (run_dir / "evaluation/final_results.json").read_text()
    )
    assert saved_evaluation == {
        "config_artifact": "config/run_config.yaml",
        "episodes": episodes,
        "scope": "final",
        "step": 100,
        "summary": summary,
    }


def test_local_logger_rejects_artifact_paths_outside_its_root(tmp_path):
    logger = LocalArtifactLogger(tmp_path / "run")

    with pytest.raises(ValueError, match="must be relative"):
        logger.log_state_dict({}, artifact_path="../outside")
