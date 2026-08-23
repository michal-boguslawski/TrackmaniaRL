import pytest
import torch as T
from typing import Callable

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep


@pytest.fixture
def rollout_buffer() -> RolloutBuffer:
    return RolloutBuffer(8, 2, (96, 96, 1), (3, ))


@pytest.fixture
def rollout_step_factory() -> Callable[[], RolloutStep]:
    def create() -> RolloutStep:
        return RolloutStep(
            observation=T.randint(
                0, 255, (2, 96, 96, 1)
            ).to(T.uint8),
            action=T.randn(2, 3).to(T.float32),
            critic_value=T.randn(2).to(T.float32),
            old_log_probs=T.rand(2).to(T.float32),
            reward=T.rand(2).to(T.float32),
            terminated=T.Tensor([False, True]).to(T.bool),
            truncated=T.Tensor([True, False]).to(T.bool),
        )

    return create


def test_rollout_buffer_shapes(rollout_buffer: RolloutBuffer):
    assert rollout_buffer._buffer["observation"].shape == (2, 8, 96, 96, 1)
    assert rollout_buffer._buffer["action"].shape == (2, 8, 3)
    assert rollout_buffer._buffer["critic_value"].shape == (2, 8)
    assert rollout_buffer._buffer["old_log_probs"].shape == (2, 8)
    assert rollout_buffer._buffer["reward"].shape == (2, 8)
    assert rollout_buffer._buffer["terminated"].shape == (2, 8)
    assert rollout_buffer._buffer["truncated"].shape == (2, 8)


def test_rollout_buffer_add(rollout_buffer: RolloutBuffer, rollout_step_factory: Callable[[], RolloutStep]):
    rollout_step = rollout_step_factory()
    rollout_buffer.add(rollout_step)

    assert rollout_buffer._counter == 1
    assert rollout_buffer.is_full() == False
    T.testing.assert_close(
        rollout_buffer._buffer["observation"][:, 0],
        rollout_step.observation,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["action"][:, 0],
        rollout_step.action,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["critic_value"][:, 0],
        rollout_step.critic_value,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["old_log_probs"][:, 0],
        rollout_step.old_log_probs,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["reward"][:, 0],
        rollout_step.reward,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["terminated"][:, 0],
        rollout_step.terminated,
    )
    T.testing.assert_close(
        rollout_buffer._buffer["truncated"][:, 0],
        rollout_step.truncated,
    )


def test_rollout_buffer_is_full(rollout_buffer: RolloutBuffer, rollout_step_factory: Callable[[], RolloutStep]):
    for _ in range(8):
        rollout_buffer.add(rollout_step_factory())

    assert rollout_buffer.is_full() == True

    with pytest.raises(IndexError):
        rollout_buffer.add(rollout_step_factory())


def test_rollout_buffer_get(rollout_buffer: RolloutBuffer, rollout_step_factory: Callable[[], RolloutStep]):
    rollout_steps = [rollout_step_factory() for _ in range(8)]

    for rollout_step in rollout_steps:
        rollout_buffer.add(rollout_step)

    batch = rollout_buffer.get()
    # assert shapes
    assert batch["observation"].shape == (2, 8, 96, 96, 1)
    assert batch["action"].shape == (2, 8, 3)
    assert batch["old_log_probs"].shape == (2, 8)
    assert batch["critic_value"].shape == (2, 8)
    assert batch["returns"].shape == (2, 8)
    assert batch["advantages"].shape == (2, 8)


def test_rollout_buffer_compute_returns_and_advantages_wo_terminated(rollout_buffer: RolloutBuffer):
    reward = T.Tensor([
        [1, 2, 3], [4, 5, 6]
    ]).to(T.float32)
    critic_value = T.Tensor([
        [0.5, -0.5, -0.7], [1., -0.3, 22.]
    ]).to(T.float32)
    terminated = T.Tensor([
        [False, False, False],
        [False, False, False],
    ])
    gamma, gae_lambda = 0.99, 0.95
    returns, advantages = rollout_buffer.compute_returns_and_advantages(reward, critic_value, terminated, gamma, gae_lambda)

    delta = T.Tensor([
        [1 + gamma * (-0.5) - 0.5, 2 + gamma * (-0.7) - (-0.5)],
        [4 + gamma * (-0.3) - 1., 5 + gamma * 22. - (-0.3)],
    ])

    expected_advantages = T.Tensor([
        [0, (delta[0, 0] + gamma * gae_lambda * delta[0, 1]), delta[0, 1]],
        [0, (delta[1, 0] + gamma * gae_lambda * delta[1, 1]), delta[1, 1]],
    ])
    T.testing.assert_close(
        advantages, expected_advantages
    )

    expected_returns = T.zeros_like(expected_advantages)
    expected_returns[:, 1:] = expected_advantages[:, 1:] + critic_value[:, :-1]
    T.testing.assert_close(
        returns, expected_returns
    )

    assert returns.shape == (2, 3)
    assert advantages.shape == (2, 3)


def test_rollout_buffer_compute_returns_and_advantages_w_terminated(rollout_buffer: RolloutBuffer):
    reward = T.Tensor([
        [1, 2, 3], [4, 5, 6]
    ]).to(T.float32)
    critic_value = T.Tensor([
        [0.5, -0.5, -0.7], [1., -0.3, 22.]
    ]).to(T.float32)
    terminated = T.Tensor([
        [False, True, False],
        [True, False, False],
    ])
    gamma, gae_lambda = 0.99, 0.95
    returns, advantages = rollout_buffer.compute_returns_and_advantages(reward, critic_value, terminated, gamma, gae_lambda)

    delta = T.Tensor([
        [1 + gamma * (-0.5) - 0.5, 2 - (-0.5)],
        [4 - 1., 5 + gamma * 22. - (-0.3)],
    ])

    expected_advantages = T.Tensor([
        [0, (delta[0, 0] + gamma * gae_lambda * delta[0, 1]), delta[0, 1]],
        [0, delta[1, 0], delta[1, 1]],
    ])
    T.testing.assert_close(
        advantages, expected_advantages
    )

    expected_returns = T.zeros_like(expected_advantages)
    expected_returns[:, 1:] = expected_advantages[:, 1:] + critic_value[:, :-1]
    T.testing.assert_close(
        returns, expected_returns
    )
