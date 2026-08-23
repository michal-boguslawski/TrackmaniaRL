import pytest
import torch as T

from rl_lib.agent import Agent


@pytest.fixture
def agent():
    return Agent(1, 3, 4)


def test_agent_preprocess_observation(agent: Agent):
    obs = T.randint(0, 256, (1, 96, 96, 1), dtype=T.uint8)
    result = agent._preprocess_observation(obs)

    assert result.shape == (1, 1, 96, 96)
    assert T.all(result >= -1.)
    assert T.all(result <= 1.)

def test_agent_feature_extract(agent: Agent):
    obs = T.randint(0, 256, (1, 96, 96, 1), dtype=T.uint8)
    features = agent.feature_extract(obs)
    assert features.shape == (1, 256)

def test_agent_get_features_window(agent: Agent):
    features = T.rand(3, 256)
    result = agent._get_features_window(features)
    assert result.shape == (3, 4, 256)
    assert T.allclose(
        T.stack([features for _ in range(4)], dim=1),
        result,
        atol=1e-6
    )

def test_agent_get_mask_window(agent: Agent):
    done = T.zeros(3, dtype=T.bool)
    result = agent._get_mask_window(done)
    assert result.shape == (3, 4)
    assert not T.any(result)

    done = T.ones(3, dtype=T.bool)
    result = agent._get_mask_window(done)
    assert result.shape == (3, 4)
    assert T.all(result[:, :-1])

    dones = [
        T.tensor([False, False, False]),
        T.tensor([False, True, False]),
        T.tensor([True, False, False]),
        T.tensor([False, True, True]),
    ]
    for done in dones:
        result = agent._get_mask_window(done)
    assert result.shape == (3, 4)
    assert (result == T.tensor([
        [True, True, False, False],
        [True, False, True,  False],
        [True, True,  True,  False],
    ])).all()

def test_temporal_encode(agent):
    features = T.rand(3, 4, 256)
    mask = T.tensor([
        [True, True, False, False],
        [True, False, True,  False],
        [True, True,  True,  False],
    ])
    masked_features = features.masked_fill(mask.unsqueeze(-1), 0.)
    masked_indices = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert all((masked_features[i, j] == 0).all() for i, j in masked_indices)

    encoded_features = agent.temporal_encode(features, mask)
    assert encoded_features.shape == (3, 256)

def test_agent_act(agent: Agent):
    obs = T.randint(0, 256, (2, 96, 96, 1), dtype=T.uint8)
    done = T.zeros(2, dtype=T.bool)
    action, log_prob, entropy = agent.act(obs, done)
    assert action.shape == (2, 3)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)

