"""Unit tests for the composed policy/value network.

`Network` is CNN -> TemporalCNN1D -> (Beta actor, value critic); the shape flow
between the three is the thing most likely to break when a config changes, so
`describe_shapes` and `config` get covered too.
"""

import pytest
import torch as T
from torch.distributions import Distribution, TransformedDistribution

from rl_lib.networks.factory import Network
from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    ConvLayerConfig,
    CriticConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)


BATCH = 2
STACK_SIZE = 4


@pytest.fixture
def network() -> Network:
    return Network(observation_dim=1, action_dim=3, stack_size=STACK_SIZE, config=NetworkConfig())


def test_network_feature_extract_shape(network: Network):
    obs = T.randn(BATCH, 1, 96, 96)

    assert network.feature_extract(obs).shape == (BATCH, 256)


def test_network_temporal_encode_shape(network: Network):
    x = T.randn(BATCH, STACK_SIZE, 256)

    assert network.temporal_encode(x).shape == (BATCH, 256)


def test_network_heads_shape(network: Network):
    x = T.randn(BATCH, 256)

    action_dist, action_mean, value = network.heads(x, 1.0)

    assert isinstance(action_dist, Distribution)
    assert action_dist.sample().shape == (BATCH, 3)
    assert action_mean.shape == (BATCH, 3)
    assert value.shape == (BATCH,)


def test_heads_at_zero_temperature_uses_the_deterministic_actor(network: Network):
    x = T.randn(BATCH, 256)

    _, mean_deterministic, value_deterministic = network.heads(x, 0.0)
    _, mean_untempered, value_untempered = network.heads(x, 1.0)

    T.testing.assert_close(mean_deterministic, mean_untempered, atol=0, rtol=0)
    T.testing.assert_close(value_deterministic, value_untempered, atol=0, rtol=0)


def test_heads_uses_the_beta_actor(network: Network):
    action_dist, _, _ = network.heads(T.randn(BATCH, 256), 1.0)

    assert isinstance(action_dist, TransformedDistribution)
    assert action_dist.base_dist.batch_shape == (BATCH, 3)


def test_value_head_is_shared_by_both_head_paths(network: Network):
    x = T.randn(BATCH, 256)

    _, _, value = network.heads(x, 0.5)
    T.testing.assert_close(value, network.critic(x))


def test_network_is_fully_differentiable(network: Network):
    obs = T.randn(BATCH * STACK_SIZE, 1, 96, 96)
    features = network.feature_extract(obs)
    temporal = network.temporal_encode(features.unfold(0, STACK_SIZE, STACK_SIZE).permute(0, 2, 1))
    _, action_mean, value = network.heads(temporal, 1.0)

    (action_mean.sum() + value.sum()).backward()

    for name, param in network.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert T.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_submodules_are_the_four_expected_modules(network: Network):
    assert [name for name, _ in network._submodules()] == [
        "cnn",
        "sequence_encoder",
        "actor",
        "critic",
    ]


def test_dims_are_wired_from_the_config():
    config = NetworkConfig(
        cnn=CNNConfig(
            conv_layers=[ConvLayerConfig(out_channels=2, kernel_size=3)],
            hidden_layers=[LinearLayerConfig(out_dim=8)],
            out_dim=4,
        ),
        temporal=TemporalConfig(out_dim=6),
        actor=ActorConfig(hidden_layers=[LinearLayerConfig(out_dim=5, activation="gelu")]),
        critic=CriticConfig(hidden_layers=[LinearLayerConfig(out_dim=7, activation="gelu")]),
    )
    network = Network(observation_dim=1, action_dim=3, stack_size=2, config=config)

    assert network.cnn.out_dim == 4
    assert network.sequence_encoder.in_dim == 4
    assert network.sequence_encoder.out_dim == 6
    assert network.actor.in_dim == network.critic.in_dim == 6
    assert network.actor.cfg.hidden_layers[0].out_dim == 5
    assert network.critic.cfg.hidden_layers[0].out_dim == 7


def test_save_and_load_state_dict_roundtrip(tmp_path, network: Network):
    path = str(tmp_path / "network.pt")
    network.save_state_dict(path)
    expected = {k: v.clone() for k, v in network.state_dict().items()}

    assert (tmp_path / "network.pt").is_file()

    with T.no_grad():
        for param in network.parameters():
            param.add_(1.0)
    network.load_state_dict(path)

    for key, value in network.state_dict().items():
        T.testing.assert_close(value, expected[key])


def test_load_state_dict_rejects_mismatched_checkpoints(tmp_path, network: Network):
    other = Network(
        observation_dim=1,
        action_dim=3,
        stack_size=STACK_SIZE,
        config=NetworkConfig(temporal=TemporalConfig(out_dim=128)),
    )
    path = str(tmp_path / "other.pt")
    other.save_state_dict(path)

    with pytest.raises(RuntimeError):
        network.load_state_dict(path)


def test_config_is_flat_and_counts_parameters(network: Network):
    config = network.config()
    total = sum(p.numel() for p in network.parameters())

    assert all("." in key or key.startswith("total") for key in config)
    assert config["cnn.out_dim"] == 256
    assert config["temporal.out_dim"] == 256
    assert '"out_dim": 256' in config["actor.hidden_layers"]
    assert '"out_dim": 256' in config["critic.hidden_layers"]
    assert config["total_params"] == total
    assert config["total_trainable_params"] == total


def test_describe_shapes_reports_the_dimension_flow(network: Network):
    report = network.describe_shapes()

    assert "observation_dim=1" in report
    assert "cnn.out_dim=256" in report
    assert f"TemporalCNN1D, stack_size={STACK_SIZE}" in report
    assert "sequence_encoder.out_dim=256" in report
    assert "action_dim=3" in report
    assert "SHAPE MISMATCHES" not in report


def test_describe_shapes_flags_a_broken_dimension_chain(network: Network):
    network.actor.in_dim = 999

    assert "SHAPE MISMATCHES DETECTED" in network.describe_shapes()
    assert "actor.in_dim=999" in network.describe_shapes()


def test_repr_lists_submodules_and_parameter_count(network: Network):
    text = repr(network)
    total = sum(p.numel() for p in network.parameters())

    assert text.startswith("Network(")
    assert "(cnn):" in text and "(critic):" in text
    assert f"total_params={total:,}" in text
