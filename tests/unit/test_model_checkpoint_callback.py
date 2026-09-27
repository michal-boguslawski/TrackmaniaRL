from types import SimpleNamespace

import torch as T

from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.model_checkpoint import ModelCheckpointCallback


class _RecordingLogger(MetricsLogger):
    def __init__(self):
        self.logged_models = []
        self.logged_state_dicts = []

    def log_metrics(self, metrics, step):
        pass

    def log_model(self, model, artifact_path="model"):
        self.logged_models.append((model, artifact_path))

    def log_state_dict(self, state_dict, artifact_path="checkpoints"):
        self.logged_state_dicts.append((state_dict, artifact_path))


def test_periodic_checkpoint_uses_logger_and_rollout_step_interval():
    state_dict = {"weight": T.tensor([1.0])}

    class Network:
        def state_dict(self):
            return state_dict

    logger = _RecordingLogger()
    callback = ModelCheckpointCallback(
        SimpleNamespace(network=Network()), [logger], interval=3
    )

    for step in range(8):
        callback.on_env_step(step=step)

    assert logger.logged_state_dicts == [
        (state_dict, "checkpoints/step_003"),
        (state_dict, "checkpoints/step_006"),
    ]


def test_final_callback_logs_model_and_state_dict_through_loggers():
    state_dict = {"weight": T.tensor([1.0])}

    class Network:
        def state_dict(self):
            return state_dict

    network = Network()
    logger = _RecordingLogger()
    callback = ModelCheckpointCallback(SimpleNamespace(network=network), [logger])

    callback.on_rollout_end()

    assert logger.logged_models == [(network, "final_model")]
    assert logger.logged_state_dicts == [(state_dict, "final_state_dict")]


def test_periodic_checkpoint_is_disabled_without_an_interval():
    logger = _RecordingLogger()
    callback = ModelCheckpointCallback(
        SimpleNamespace(network=object()), [logger]
    )

    callback.on_env_step(step=1000)

    assert logger.logged_state_dicts == []
