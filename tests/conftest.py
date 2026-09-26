"""Shared fixtures.

The networks keep the same layer topology as production but with tiny widths
so the CPU-only test suite stays fast. Every test module gets deterministic
RNGs, otherwise sampling-based assertions flake.
"""

import numpy as np
import pytest
import torch as T

from rl_lib.agent import Agent
from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    CriticConfig,
    ConvLayerConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)
from rl_lib.networks.factory import Network
from rl_lib.run_config import AgentSettings


OBSERVATION_SHAPE = (96, 96, 1)
OBSERVATION_DIM = 1
ACTION_DIM = 3
STACK_SIZE = 2
FEATURE_DIM = 4


@pytest.fixture(autouse=True)
def deterministic_rngs():
    T.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def network_config() -> NetworkConfig:
    return NetworkConfig(
        cnn=CNNConfig(
            conv_layers=[
                ConvLayerConfig(out_channels=2, kernel_size=8, stride=4),
                ConvLayerConfig(out_channels=4, kernel_size=4, stride=2),
                ConvLayerConfig(out_channels=8, kernel_size=3),
                ConvLayerConfig(out_channels=8, kernel_size=3, stride=2),
            ],
            hidden_layers=[LinearLayerConfig(out_dim=8)],
            out_dim=FEATURE_DIM,
        ),
        temporal=TemporalConfig(out_dim=FEATURE_DIM),
        actor=ActorConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
        critic=CriticConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
    )


@pytest.fixture
def network(network_config: NetworkConfig) -> Network:
    return Network(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        stack_size=STACK_SIZE,
        config=network_config,
    )


@pytest.fixture
def agent_settings() -> AgentSettings:
    return AgentSettings(stack_size=STACK_SIZE)


@pytest.fixture
def agent(network: Network, agent_settings: AgentSettings) -> Agent:
    return Agent(network=network, device="cpu", config=agent_settings)


@pytest.fixture
def observations():
    """Factory for uint8 NHWC observations of shape (batch, 96, 96, 1)."""

    def make(batch: int = 2, fill: int | None = None) -> T.Tensor:
        if fill is None:
            return T.randint(0, 256, (batch, *OBSERVATION_SHAPE), dtype=T.uint8)
        return T.full((batch, *OBSERVATION_SHAPE), fill, dtype=T.uint8)

    return make


@pytest.fixture
def dones():
    """Factory for per-environment done flags."""

    def make(batch: int = 2, value: bool = False) -> T.Tensor:
        return T.full((batch,), value, dtype=T.bool)

    return make
