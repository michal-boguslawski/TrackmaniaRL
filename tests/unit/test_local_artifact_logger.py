import pytest
import torch as T

from rl_lib.tracking.local_artifact_logger import LocalArtifactLogger


def test_local_logger_persists_state_dict_and_model_under_artifact_paths(tmp_path):
    logger = LocalArtifactLogger(tmp_path / "run")
    model = T.nn.Linear(2, 1)
    state_dict = model.state_dict()

    logger.log_state_dict(state_dict, artifact_path="checkpoints/step_20")
    logger.log_state_dict(state_dict, artifact_path="final_state_dict")
    logger.log_model(model, artifact_path="final_model")

    checkpoint = T.load(
        tmp_path / "run" / "checkpoints" / "step_20.pt",
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


def test_local_logger_rejects_artifact_paths_outside_its_root(tmp_path):
    logger = LocalArtifactLogger(tmp_path / "run")

    with pytest.raises(ValueError, match="must be relative"):
        logger.log_state_dict({}, artifact_path="../outside")
