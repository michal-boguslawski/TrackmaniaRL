"""Unit tests for the network building blocks.

`Network` is exercised in `test_network.py`; this module covers the submodules
it composes, the pydantic configs that parameterise them and the weight
initialisation helper.
"""

import math

import pytest
import torch as T
from torch import nn

from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    ConvLayerConfig,
    CriticConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)
from rl_lib.networks.critic import Critic
from rl_lib.networks.cnn import CNN
from rl_lib.networks.temporal import TemporalCNN1D
from rl_lib.networks.utils import init_layer


OBSERVATION_DIM = 1
FEATURE_DIM = 4
STACK_SIZE = 2


@pytest.fixture
def cnn() -> CNN:
    return CNN(OBSERVATION_DIM, config=CNNConfig(
        conv_layers=[
            ConvLayerConfig(out_channels=2, kernel_size=8, stride=4),
            ConvLayerConfig(out_channels=4, kernel_size=4, stride=2),
            ConvLayerConfig(out_channels=8, kernel_size=3),
            ConvLayerConfig(out_channels=8, kernel_size=3, stride=2),
        ],
        hidden_layers=[LinearLayerConfig(out_dim=8)],
        out_dim=FEATURE_DIM,
    ))


@pytest.fixture
def temporal() -> TemporalCNN1D:
    return TemporalCNN1D(STACK_SIZE, in_dim=FEATURE_DIM, config=TemporalConfig(out_dim=FEATURE_DIM))


@pytest.fixture
def critic() -> Critic:
    return Critic(
        in_dim=FEATURE_DIM,
        config=CriticConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
    )


# ---------------------------------------------------------------- CNN


def test_cnn_maps_an_observation_stack_to_features(cnn: CNN):
    batch = T.zeros(3, OBSERVATION_DIM, 96, 96)

    features = cnn(batch)

    assert features.shape == (3, FEATURE_DIM)


def test_cnn_output_dim_property(cnn: CNN):
    assert cnn.out_dim == FEATURE_DIM


def test_cnn_is_deterministic_in_eval_and_differentiable(cnn: CNN):
    batch = T.rand(2, OBSERVATION_DIM, 96, 96)
    cnn.eval()

    with T.no_grad():
        first = cnn(batch)
        second = cnn(batch)
    T.testing.assert_close(first, second)

    cnn.train()
    cnn(batch).sum().backward()
    assert all(p.grad is not None and T.isfinite(p.grad).all() for p in cnn.parameters())


def test_cnn_adapts_to_configured_input_sizes(cnn: CNN):
    assert cnn(T.zeros(1, OBSERVATION_DIM, 64, 64)).shape == (1, FEATURE_DIM)


def test_cnn_config_reports_dims_and_param_count(cnn: CNN):
    config = cnn.config()

    assert config["observation_dim"] == OBSERVATION_DIM
    assert "conv_layers" in config
    assert "adaptive_pool_size" in config
    assert config["out_dim"] == FEATURE_DIM
    assert config["n_params"] == sum(p.numel() for p in cnn.parameters())


def test_cnn_repr_lists_the_config_and_the_param_count(cnn: CNN):
    text = repr(cnn)

    assert text.startswith("CNN(")
    assert "observation_dim=1" in text
    assert "out_dim=4" in text
    assert f"params={sum(p.numel() for p in cnn.parameters()):,}" in text


# ---------------------------------------------------------------- temporal


def test_temporal_collapses_the_stack_dimension(temporal: TemporalCNN1D):
    batch = T.rand(5, STACK_SIZE, FEATURE_DIM)

    encoded = temporal(batch)

    assert encoded.shape == (5, FEATURE_DIM)
    assert temporal.out_dim == FEATURE_DIM


def test_temporal_normalises_before_encoding(temporal: TemporalCNN1D):
    batch = T.rand(4, STACK_SIZE, FEATURE_DIM) * 10.0

    encoded = temporal(batch)
    manual = temporal._encoder(temporal._norm(batch).permute(0, 2, 1))
    T.testing.assert_close(encoded, temporal._out_norm(manual.permute(0, 2, 1)).squeeze(1))


def test_temporal_does_not_modify_its_input(temporal: TemporalCNN1D):
    batch = T.rand(4, STACK_SIZE, FEATURE_DIM)
    before = batch.clone()

    temporal(batch)

    T.testing.assert_close(batch, before), "forward must not squeeze the caller's tensor"


def test_temporal_handles_a_single_frame_stack():
    encoder = TemporalCNN1D(1, in_dim=FEATURE_DIM, config=TemporalConfig(out_dim=FEATURE_DIM))

    encoded = encoder(T.rand(2, 1, FEATURE_DIM))

    assert encoded.shape == (2, FEATURE_DIM)


def test_temporal_is_differentiable(temporal: TemporalCNN1D):
    temporal(T.rand(3, STACK_SIZE, FEATURE_DIM)).sum().backward()

    assert all(p.grad is not None for p in temporal.parameters())


def test_temporal_config_reports_dims_and_param_count(temporal: TemporalCNN1D):
    config = temporal.config()

    assert config["stack_size"] == STACK_SIZE
    assert config["in_dim"] == FEATURE_DIM
    assert config["out_dim"] == FEATURE_DIM
    assert config["n_params"] == sum(p.numel() for p in temporal.parameters())


def test_temporal_repr_lists_the_config_and_the_param_count(temporal: TemporalCNN1D):
    text = repr(temporal)

    assert text.startswith("TemporalCNN1D(")
    assert f"stack_size={STACK_SIZE}" in text
    assert f"in_dim={FEATURE_DIM}" in text
    assert f"params={sum(p.numel() for p in temporal.parameters()):,}" in text


# ---------------------------------------------------------------- critic


def test_critic_returns_one_value_per_sample(critic: Critic):
    values = critic(T.rand(6, FEATURE_DIM))

    assert values.shape == (6,)


def test_critic_squeezes_a_single_sample(critic: Critic):
    assert critic(T.rand(FEATURE_DIM)).shape == T.Size([])


def test_critic_is_differentiable(critic: Critic):
    critic(T.rand(3, FEATURE_DIM)).sum().backward()

    assert all(p.grad is not None for p in critic.parameters())


def test_critic_config_reports_dims_and_param_count(critic: Critic):
    config = critic.config()

    assert config["in_dim"] == FEATURE_DIM
    assert config["hidden_layers"][0]["out_dim"] == 8
    assert config["n_params"] == sum(p.numel() for p in critic.parameters())


def test_critic_repr_lists_the_config_and_the_param_count(critic: Critic):
    text = repr(critic)

    assert text.startswith("Critic(")
    assert f"in_dim={FEATURE_DIM}" in text
    assert f"params={sum(p.numel() for p in critic.parameters()):,}" in text


def test_critic_out_dim(critic: Critic):
    assert critic.out_dim == 1


# ---------------------------------------------------------------- configs


@pytest.mark.parametrize(
    "config_class",
    [
        CNNConfig,
        TemporalConfig,
        ActorConfig,
        CriticConfig,
    ],
)
def test_subconfig_defaults_match_the_production_topology(config_class):
    model = config_class()
    if config_class in (ActorConfig, CriticConfig):
        assert model.hidden_layers[0].out_dim == 256
    else:
        assert model.out_dim == 256


@pytest.mark.parametrize("value", [0, -1, 0.5])
def test_subconfigs_reject_non_positive_dims(value):
    with pytest.raises(Exception):
        CNNConfig(out_dim=value)
    with pytest.raises(Exception):
        ActorConfig(hidden_layers=[LinearLayerConfig(out_dim=value)])


def test_network_config_defaults_to_production_topology():
    cfg = NetworkConfig()

    assert [layer.out_channels for layer in cfg.cnn.conv_layers] == [32, 64, 128, 128]
    assert cfg.cnn.hidden_layers[0].out_dim == 256
    assert cfg.cnn.adaptive_pool_size == 3
    assert cfg.temporal.out_dim == 256
    assert cfg.actor.hidden_layers[0].out_dim == 256
    assert cfg.critic.hidden_layers[0].out_dim == 256


def test_network_config_sub_configs_are_not_shared_between_instances():
    first = NetworkConfig()
    second = NetworkConfig()

    first.cnn.conv_layers[0].out_channels = 8

    assert second.cnn.conv_layers[0].out_channels == 32


def test_network_config_round_trips_through_a_dict():
    cfg = NetworkConfig(
        cnn=CNNConfig(
            conv_layers=[ConvLayerConfig(out_channels=2, kernel_size=3)],
            hidden_layers=[LinearLayerConfig(out_dim=8)],
            out_dim=FEATURE_DIM,
        ),
        temporal=TemporalConfig(out_dim=FEATURE_DIM),
        actor=ActorConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
        critic=CriticConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
    )

    assert NetworkConfig(**cfg.model_dump()) == cfg


# ---------------------------------------------------------------- init helper


def test_init_layer_zeroes_the_bias_and_keeps_the_gain():
    layer = nn.Linear(16, 8)

    returned = init_layer(layer)

    assert returned is layer, "the layer is returned so it can be chained"
    T.testing.assert_close(layer.bias, T.zeros(8))


def test_init_layer_produces_orthogonal_weights():
    T.manual_seed(0)
    layer = init_layer(nn.Linear(32, 16), gain=1.0)

    # the weight is (out, in) = (16, 32), so the rows are the orthonormal ones
    product = layer.weight @ layer.weight.T
    T.testing.assert_close(product, T.eye(16), atol=1e-5, rtol=1e-5)


def test_init_layer_scales_the_norms_with_the_gain():
    T.manual_seed(0)
    layer = init_layer(nn.Linear(64, 32), gain=math.sqrt(2))

    square_norm = (layer.weight**2).sum().item() / 32

    assert square_norm == pytest.approx(2.0, rel=0.15)


def test_init_layer_works_for_convolution_layers():
    layer = init_layer(nn.Conv2d(1, 2, kernel_size=3))

    assert layer.bias.abs().max() == 0.0
    # `orthogonal_` flattens the weight, so each output filter stays unit norm
    # times the gain
    T.testing.assert_close(
        layer.weight.flatten(1).norm(dim=1),
        T.full((2,), math.sqrt(2)),
        atol=1e-5,
        rtol=1e-5,
    )
