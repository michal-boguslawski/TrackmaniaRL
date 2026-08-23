import pytest
import torch as T
from torch.distributions import Distribution

from rl_lib.networks.factory import Network


@pytest.fixture
def network():
    return Network(
        observation_dim=1,
        action_dim=3,
        stack_size=4,
    )


def test_network_feature_extract_shape(network: Network):
    obs = T.randn(2, 1, 96, 96)
    assert network.feature_extract(obs).shape == (2, 256)

def test_network_temporal_encode_shape(network: Network):
    x = T.randn(2, 4, 256)
    assert network.temporal_encode(x).shape == (2, 256)

def test_network_heads_shape(network: Network):
    x = T.randn(2, 256)
    action_dist, value = network.heads(x)
    assert isinstance(action_dist, Distribution)
    assert value.shape == (2, )
