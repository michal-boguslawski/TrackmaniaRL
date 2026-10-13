"""Tests for source-specific model checkpoint loading."""

from types import SimpleNamespace

import torch as T

from rl_lib.checkpoint_loaders import MLflowCheckpointLoader
from rl_lib.networks.factory import Network


def _make_network(network_config) -> Network:
    """Build a test network with the shared lightweight test topology."""
    return Network(
        observation_dim=1,
        action_dim=3,
        stack_size=2,
        config=network_config,
    )


def test_mlflow_loader_loads_compiled_final_model(
    monkeypatch, network_config, network: Network
):
    source = _make_network(network_config)
    expected = {key: value.clone() for key, value in source.state_dict().items()}
    calls = {}
    fake_mlflow = SimpleNamespace(
        pytorch=SimpleNamespace(
            load_model=lambda uri, map_location: calls.update(
                uri=uri, map_location=map_location
            ) or SimpleNamespace(_orig_mod=source),
        ),
    )
    monkeypatch.setattr("rl_lib.checkpoint_loaders.mlflow", fake_mlflow)

    loaded = MLflowCheckpointLoader("runs:/run-123/final_model").load(network, "cpu")

    assert loaded is network
    assert calls["uri"] == "runs:/run-123/final_model"
    for key, value in network.state_dict().items():
        T.testing.assert_close(value, expected[key])


def test_mlflow_loader_downloads_periodic_state_dict(
    monkeypatch, tmp_path, network_config, network: Network
):
    source = _make_network(network_config)
    expected = {key: value.clone() for key, value in source.state_dict().items()}
    checkpoint_path = tmp_path / "checkpoint.pth"
    T.save(
        {"state_dict": {f"_orig_mod.{key}": value for key, value in expected.items()}},
        checkpoint_path,
    )
    calls = {}
    fake_mlflow = SimpleNamespace(
        artifacts=SimpleNamespace(
            download_artifacts=lambda **kwargs: calls.update(kwargs)
            or str(checkpoint_path),
        ),
        pytorch=SimpleNamespace(
            load_model=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("periodic checkpoints are state dicts, not models")
            ),
        ),
    )
    monkeypatch.setattr("rl_lib.checkpoint_loaders.mlflow", fake_mlflow)

    loaded = MLflowCheckpointLoader("run-123", checkpoint_step=500).load(network, "cpu")

    assert loaded is network
    assert calls == {
        "run_id": "run-123",
        "artifact_path": "checkpoints/step_500/state_dict.pth",
    }
    for key, value in network.state_dict().items():
        T.testing.assert_close(value, expected[key])
