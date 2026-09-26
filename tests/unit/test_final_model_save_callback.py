from types import SimpleNamespace

import torch as T

from rl_lib.training.callbacks.final_model_save import FinalModelSaveCallback


def test_final_model_callback_saves_model_and_logs_state_dict(tmp_path, monkeypatch):
    state_dict = {"weight": T.tensor([1.0])}

    class Network:
        def state_dict(self):
            return state_dict

    class MLflowLogger:
        def __init__(self):
            self.logged_models = []
            self.logged_state_dicts = []

        def log_model(self, model, artifact_path):
            self.logged_models.append((model, artifact_path))

        def log_state_dict(self, state_dict, artifact_path):
            self.logged_state_dicts.append((state_dict, artifact_path))

    saved_models = []
    monkeypatch.setattr(T, "save", lambda model, path: saved_models.append((model, path)))
    network = Network()
    mlflow_logger = MLflowLogger()
    callback = FinalModelSaveCallback(
        SimpleNamespace(network=network),
        tmp_path / "run",
        mlflow_logger,
    )

    callback.on_rollout_end()

    model_path = tmp_path / "run" / "final_model.pt"
    assert saved_models == [(network, model_path)]
    assert mlflow_logger.logged_models == [(network, "final_model")]
    assert mlflow_logger.logged_state_dicts == [(state_dict, "final_state_dict")]
