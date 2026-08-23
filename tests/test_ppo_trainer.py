import pytest
import torch as T

from rl_lib.training.ppo_trainer import PPOTrainer
from rl_lib.agent import Agent


@pytest.fixture
def agent():
    return Agent(1, 3, 4)


@pytest.fixture
def trainer(agent: Agent):
    return PPOTrainer(agent)


def get_batch():
    return {
        "observation": T.rand(2, 35, 96, 96, 1),
        "action": T.rand(2, 32),
        "old_log_probs": T.rand(2, 32),
        "old_values": T.rand(2, 32),
        "returns": T.rand(2, 32),
        "advantages": T.rand(2, 32),
        "dones": T.zeros(2, 35).to(T.bool),
    }


def test_trainer_get_iid_minibatches_shapes(trainer: PPOTrainer):
    batch = get_batch()
    minibatches = trainer._get_iid_minibatches(batch, 8, 4)
    minibatch = next(minibatches)
    assert minibatch["observation"].shape == (11, 96, 96, 1)
    assert minibatch["action"].shape == (8, )
    assert minibatch["old_log_probs"].shape == (8, )
    assert minibatch["old_values"].shape == (8, )
    assert minibatch["returns"].shape == (8, )
    assert minibatch["advantages"].shape == (8, )
    assert minibatch["dones"].shape == (11, )

def test_trainer_get_iid_minibatches_counts(trainer: PPOTrainer):
    batch = get_batch()
    minibatches = trainer._get_iid_minibatches(batch, 8, 4)
    for i, _ in enumerate(minibatches):
        pass
    assert i == 7
