"""Unit tests for preallocated rollout storage and advantage calculation.

The observation and episode-boundary storage are `stack_size - 1` columns
longer than the transition storage because the PPO update rebuilds a temporal
window per transition; `get` therefore returns `size` observation columns
against `size - 1` transitions.

GAE keeps Gymnasium's distinction: a truncation bootstraps from the value of the
final observation, a termination does not, and the recursion stops at either.
"""

import dataclasses

import pytest
import torch as T

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.run_config import RolloutSettings


SIZE = 4
STACK_SIZE = 2
NUM_ENVS = 2
OBSERVATION_SHAPE = (96, 96, 1)
ACTION_DIM = 3

GAMMA = 0.9
LAMBDA = 0.5


@pytest.fixture
def buffer() -> RolloutBuffer:
    return RolloutBuffer(
        RolloutSettings(buffer_size=SIZE, gamma=GAMMA, gae_lambda=LAMBDA),
        stack_size=STACK_SIZE,
    )


@pytest.fixture
def step_factory():
    """Builds a `RolloutStep` for a whole vectorised batch of environments."""

    def make(
        index: int = 0,
        reward: float = 1.0,
        terminated: T.Tensor | None = None,
        truncated: T.Tensor | None = None,
        truncated_value: T.Tensor | None = None,
        observation_value: int = 0,
        critic_value: float = 0.0,
    ) -> RolloutStep:
        return RolloutStep(
            observation=T.full((NUM_ENVS, *OBSERVATION_SHAPE), observation_value, dtype=T.uint8),
            action=T.full((NUM_ENVS, ACTION_DIM), float(index)),
            critic_value=T.full((NUM_ENVS,), critic_value),
            old_log_probs=T.full((NUM_ENVS, ACTION_DIM), float(index)),
            reward=T.full((NUM_ENVS,), reward),
            terminated=terminated if terminated is not None else T.zeros(NUM_ENVS, dtype=T.bool),
            truncated=truncated if truncated is not None else T.zeros(NUM_ENVS, dtype=T.bool),
            truncated_value=truncated_value,
        )

    return make


def fill_buffer(buffer: RolloutBuffer, step_factory) -> None:
    for index in range(SIZE):
        buffer.add(
            step_factory(
                index=index,
                reward=float(index),
                observation_value=index,
                critic_value=float(index),
            )
        )


def test_buffer_starts_empty(buffer: RolloutBuffer):
    assert buffer.size == SIZE
    assert buffer._counter == 0
    assert buffer.is_full() is False
    assert buffer._buffer == {}
    assert f"counter=0/{SIZE}" in repr(buffer)


def test_add_stores_every_field_and_counts_the_step(buffer: RolloutBuffer, step_factory):
    step = step_factory(index=3, terminated=T.tensor([True, False]), truncated=T.tensor([False, True]))

    buffer.add(step)

    assert buffer._counter == 1
    assert buffer.is_full() is False
    for key, expected in (
        ("action", step.action),
        ("critic_value", step.critic_value),
        ("old_log_probs", step.old_log_probs),
        ("reward", step.reward),
    ):
        T.testing.assert_close(buffer._ordered(key)[:, 0], expected)

    T.testing.assert_close(
        buffer._ordered("terminated", windowed=True)[:, -1], step.terminated
    )
    T.testing.assert_close(
        buffer._ordered("truncated", windowed=True)[:, -1], step.truncated
    )


def test_add_stores_copies_and_never_aliases_the_step(buffer: RolloutBuffer, step_factory):
    step = step_factory()
    buffer.add(step)

    step.reward.fill_(99.0)

    assert buffer._ordered("reward")[0, 0].item() == 1.0, "the buffer must not alias the caller's tensor"


def test_add_reuses_preallocated_storage(buffer: RolloutBuffer, step_factory):
    buffer.add(step_factory(index=0))
    storage_ids = {key: id(storage) for key, storage in buffer._buffer.items()}

    buffer.add(step_factory(index=1))

    assert {key: id(storage) for key, storage in buffer._buffer.items()} == storage_ids


def test_add_pads_the_window_buffers_with_the_first_step(buffer: RolloutBuffer, step_factory):
    """`stack_size - 1` copies of the first step let the update slice a full
    observation window out of a single stored state."""

    first = step_factory(observation_value=7, terminated=T.tensor([True, False]), truncated=T.tensor([False, True]))
    buffer.add(first)

    assert buffer._window_count == STACK_SIZE
    observations = buffer._ordered("observation", windowed=True)
    terminated = buffer._ordered("terminated", windowed=True)
    for index in range(STACK_SIZE):
        T.testing.assert_close(observations[:, index], first.observation)
        T.testing.assert_close(terminated[:, index], first.terminated)

    # the non-windowed fields are stored exactly once per step
    for key in ("action", "critic_value", "old_log_probs", "reward", "truncated_value"):
        assert buffer._ordered(key).shape[1] == 1


def test_padding_only_happens_at_the_start_of_a_buffer(buffer: RolloutBuffer, step_factory):
    buffer.add(step_factory(observation_value=1))
    buffer.add(step_factory(observation_value=2))

    assert buffer._window_count == STACK_SIZE + 1
    ordered = buffer._ordered("observation", windowed=True)
    T.testing.assert_close(ordered[0, 0, 0, 0, 0], T.tensor(1, dtype=T.uint8))
    T.testing.assert_close(ordered[0, -1, 0, 0, 0], T.tensor(2, dtype=T.uint8))


def test_is_full_after_exactly_size_steps(buffer: RolloutBuffer, step_factory):
    for index in range(SIZE):
        assert buffer.is_full() is (index == SIZE)
        buffer.add(step_factory(index=index))

    assert buffer.is_full() is True


def test_add_past_capacity_keeps_the_newest_transitions(buffer: RolloutBuffer, step_factory):
    for index in range(SIZE + 2):
        buffer.add(step_factory(index=index, observation_value=index))

    assert buffer._transition_count == SIZE
    T.testing.assert_close(
        buffer._ordered("action")[:, -1, 0],
        T.full((NUM_ENVS,), float(SIZE + 1)),
    )
    assert buffer._window_count == SIZE + STACK_SIZE - 1

    batch = buffer.get()
    T.testing.assert_close(
        batch["action"][0, :, 0],
        T.tensor([float(index) for index in range(2, SIZE + 1)]),
    )
    assert batch["observation"][0, :, 0, 0, 0].tolist() == [1, 2, 3, 4]


def test_reset_counter_keeps_the_stored_transitions(buffer: RolloutBuffer, step_factory):
    fill_buffer(buffer, step_factory)

    buffer.reset_counter()

    assert buffer._counter == 0
    assert buffer.is_full() is False
    assert buffer._transition_count == SIZE


def test_reset_clears_every_field(buffer: RolloutBuffer, step_factory):
    fill_buffer(buffer, step_factory)

    buffer.reset()

    assert buffer._counter == 0
    assert buffer._buffer == {}


def test_add_defaults_a_missing_truncated_value_to_zeros(buffer: RolloutBuffer, step_factory):
    buffer.add(step_factory(reward=2.0))

    T.testing.assert_close(buffer._ordered("truncated_value")[:, 0], T.zeros(NUM_ENVS))


def test_rollout_step_is_an_immutable_record(step_factory):
    step = step_factory()

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.reward = T.zeros(NUM_ENVS)


def test_config_reports_size_and_stack_size(buffer: RolloutBuffer):
    assert buffer.config() == {
        "size": SIZE,
        "stack_size": STACK_SIZE,
        "gamma": GAMMA,
        "gae_lambda": LAMBDA,
    }


def test_get_returns_one_column_less_for_transitions_than_for_observations(buffer: RolloutBuffer, step_factory):
    fill_buffer(buffer, step_factory)

    batch = buffer.get()

    assert batch["observation"].shape == (NUM_ENVS, SIZE, *OBSERVATION_SHAPE)
    assert batch["dones"].shape == (NUM_ENVS, SIZE)
    assert batch["action"].shape == (NUM_ENVS, SIZE - 1, ACTION_DIM)
    assert batch["old_log_probs"].shape == (NUM_ENVS, SIZE - 1, ACTION_DIM)
    assert batch["critic_value"].shape == (NUM_ENVS, SIZE - 1)
    assert batch["returns"].shape == (NUM_ENVS, SIZE - 1)
    assert batch["advantages"].shape == (NUM_ENVS, SIZE - 1)


def test_get_keeps_the_transition_order(buffer: RolloutBuffer, step_factory):
    fill_buffer(buffer, step_factory)

    batch = buffer.get()

    for index in range(SIZE - 1):
        T.testing.assert_close(batch["action"][:, index, 0], T.full((NUM_ENVS,), float(index)))
        T.testing.assert_close(batch["old_log_probs"][:, index, 0], T.full((NUM_ENVS,), float(index)))
        T.testing.assert_close(batch["critic_value"][:, index], T.full((NUM_ENVS,), float(index)))


def test_get_returns_a_window_of_observations_per_transition(buffer: RolloutBuffer, step_factory):
    """Transition `k` is paired with the states of steps `k - stack_size + 1..k`,
    and step 0 is duplicated so the first transition still has a full window."""

    for index in range(SIZE):
        buffer.add(step_factory(index=index, observation_value=index))

    batch = buffer.get(gamma=0.0, gae_lambda=0.0)

    for transition in range(SIZE - 1):
        window = batch["observation"][0, transition : transition + STACK_SIZE, 0, 0, 0]
        # the very first transition sees step 0 twice, later ones see the two
        # most recent states
        expected = [0, 0] if transition == 0 else [transition - 1, transition]
        assert window.tolist() == expected, f"window for transition {transition}"


def test_get_marks_episode_starts_on_the_observation_columns(buffer: RolloutBuffer, step_factory):
    """`dones` means "this frame is the first of a new episode", the convention
    `Agent.act` masks with. A step stores the flag produced by stepping *at* it,
    so its boundary belongs to the following frame - and therefore to the
    following observation column, since column `c` is step `c - stack_size + 1`.
    """

    for index in range(SIZE):
        buffer.add(
            step_factory(
                index=index,
                terminated=T.tensor([index == 1, False]),
                truncated=T.tensor([False, index == 1]),
            )
        )

    batch = buffer.get()

    # the step-1 boundary is the start of step 2, which is column 3
    assert [bool(v) for v in batch["dones"][0]] == [False, False, False, True]
    assert [bool(v) for v in batch["dones"][1]] == [False, False, False, True]


def test_done_windows_mask_the_frames_from_before_the_boundary(buffer: RolloutBuffer, step_factory):
    """The window the PPO update masks must keep the frames of the episode that
    is running and drop the ones left over from the episode before it - the
    inverse is what a misaligned flag produces."""

    for index in range(SIZE):
        buffer.add(step_factory(index=index, terminated=T.tensor([index == 1, False])))

    batch = buffer.get()

    def mask_for(transition: int) -> list[bool]:
        window = batch["dones"][0, transition : transition + STACK_SIZE]
        kept = window.logical_not() & (window.flip(0).cumsum(0).flip(0) > 0)
        return kept.tolist()

    # the boundary ends step 1, so step 2 opens a new episode: its window keeps
    # step 2 and drops the leftover step 1, while earlier windows are untouched
    assert mask_for(0) == [False, False]
    assert mask_for(1) == [False, False]
    assert mask_for(2) == [True, False]


def test_get_uses_the_gamma_and_lambda_it_is_given(buffer: RolloutBuffer, step_factory):
    for index in range(SIZE):
        buffer.add(step_factory(index=index, reward=1.0, critic_value=0.5))

    discounted = buffer.get(gamma=0.9, gae_lambda=0.5)
    undiscounted = buffer.get(gamma=0.0, gae_lambda=1.0)

    # with gamma = 0 the advantage is the TD(0) residual
    T.testing.assert_close(undiscounted["advantages"], T.full((NUM_ENVS, SIZE - 1), 0.5))
    assert (discounted["advantages"] > 0).all(), "positive rewards with a positive value give positive advantages"
    T.testing.assert_close(discounted["returns"], discounted["advantages"] + discounted["critic_value"])


def test_get_bootstraps_a_truncated_step_from_its_final_observation_value(buffer: RolloutBuffer, step_factory):
    for index in range(SIZE):
        truncated = T.tensor([index == 1, False])
        buffer.add(
            step_factory(
                index=index,
                reward=1.0,
                critic_value=0.5,
                truncated=truncated,
                truncated_value=T.where(truncated, T.full((NUM_ENVS,), 7.0), T.zeros(NUM_ENVS)),
            )
        )

    batch = buffer.get(gamma=0.9, gae_lambda=0.5)

    # the last step is not truncated, so the bootstrap value cannot leak into it
    T.testing.assert_close(batch["advantages"][0, SIZE - 2], T.tensor(1.0 + 0.9 * 0.5 - 0.5))
    assert batch["advantages"][0, 0] > batch["advantages"][0, SIZE - 2]


def test_get_ignores_the_truncated_value_of_a_terminated_step(buffer: RolloutBuffer, step_factory):
    for index in range(SIZE):
        terminated = T.tensor([index == 0, False])
        buffer.add(
            step_factory(
                index=index,
                reward=1.0,
                critic_value=0.5,
                terminated=terminated,
                truncated=T.tensor([index == 0, False]),
                truncated_value=T.full((NUM_ENVS,), 100.0),
            )
        )

    batch = buffer.get(gamma=0.9, gae_lambda=0.5)

    # termination removes the bootstrap entirely, the 100.0 must not show up
    T.testing.assert_close(batch["advantages"][0, 0], T.tensor(1.0 - 0.5))

    # env 1 never ends an episode: the advantage keeps accumulating the constant
    # TD(0) residual 1 + 0.9 * 0.5 - 0.5 = 0.95
    residual = 1.0 + 0.9 * 0.5 - 0.5
    expected = residual * (1.0 + 0.9 * 0.5 + (0.9 * 0.5) ** 2)
    T.testing.assert_close(batch["advantages"][1, 0], T.tensor(expected))


def _gae_reference(
    reward: T.Tensor,
    critic_value: T.Tensor,
    terminated: T.Tensor,
    truncated: T.Tensor,
    gamma: float,
    gae_lambda: float,
    truncated_value: T.Tensor | None = None,
) -> tuple[T.Tensor, T.Tensor]:
    """Straightforward per-step reference implementation of the estimator."""

    length = reward.shape[1]
    advantages = T.zeros_like(reward)
    running = T.zeros(reward.shape[0])
    for step in reversed(range(length)):
        next_value = critic_value[:, step + 1]
        if truncated_value is not None:
            next_value = T.where(truncated[:, step], truncated_value[:, step], next_value)
        bootstrap = T.logical_not(terminated[:, step]).to(reward.dtype)
        delta = reward[:, step] + gamma * next_value * bootstrap - critic_value[:, step]
        keep = T.logical_not(T.logical_or(terminated[:, step], truncated[:, step])).to(reward.dtype)
        running = delta + gamma * gae_lambda * running * keep
        advantages[:, step] = running
    return advantages + critic_value[:, :length], advantages


@pytest.mark.parametrize("gamma,gae_lambda", [(0.9, 0.5), (0.99, 0.95), (0.0, 1.0)])
@pytest.mark.parametrize("length", [5, 17, 64])
def test_returns_and_advantages_match_the_reference_implementation(
    gamma: float, gae_lambda: float, length: int
):
    reward = T.rand(3, length)
    critic_value = T.rand(3, length + 1)
    terminated = T.zeros(3, length, dtype=T.bool)
    truncated = T.zeros(3, length, dtype=T.bool)
    terminated[0, 2] = True
    truncated[1, 3] = True
    truncated[2, 0] = True
    truncated_value = T.rand(3, length)

    returns, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward, critic_value, terminated, truncated, gamma, gae_lambda, truncated_value
    )
    expected_returns, expected_advantages = _gae_reference(
        reward, critic_value, terminated, truncated, gamma, gae_lambda, truncated_value
    )

    T.testing.assert_close(advantages, expected_advantages)
    T.testing.assert_close(returns, expected_returns)
    assert T.allclose(returns, advantages + critic_value[:, :-1])


def test_returns_and_advantages_without_any_episode_boundary():
    reward = T.tensor([[1.0, 2.0, 3.0]])
    critic_value = T.tensor([[0.5, -0.5, 0.25, 2.0]])
    no_dones = T.zeros(1, 3, dtype=T.bool)

    returns, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward, critic_value, no_dones, no_dones, gamma=GAMMA, gae_lambda=LAMBDA
    )

    T.testing.assert_close(
        advantages,
        T.tensor([[0.05 + 0.45 * (2.725 + 0.45 * 4.55), 2.725 + 0.45 * 4.55, 4.55]]),
    )
    T.testing.assert_close(returns, advantages + critic_value[:, :-1])


def test_termination_stops_the_bootstrap_and_the_recursion():
    reward = T.tensor([[1.0, 2.0, 3.0]])
    critic_value = T.tensor([[0.5, -0.5, 0.25, 2.0]])
    terminated = T.tensor([[False, True, False]])
    truncated = T.zeros(1, 3, dtype=T.bool)

    _, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward, critic_value, terminated, truncated, gamma=GAMMA, gae_lambda=LAMBDA
    )

    # no bootstrap through critic_value[:, 2] and no recursion into the future
    T.testing.assert_close(advantages, T.tensor([[0.05 + 0.45 * 2.5, 2.5, 4.55]]))


def test_truncation_bootstraps_from_the_final_observation_value():
    reward = T.tensor([[1.0]])
    critic_value = T.tensor([[0.5, 100.0]])  # the 100.0 belongs to the autoreset observation
    terminated = T.tensor([[False]])
    truncated = T.tensor([[True]])
    final_observation_value = T.tensor([[3.0]])

    returns, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward,
        critic_value,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.95,
        truncated_value=final_observation_value,
    )

    expected_return = T.tensor([[1.0 + 0.9 * 3.0]])
    T.testing.assert_close(returns, expected_return)
    T.testing.assert_close(advantages, expected_return - critic_value[:, :-1])


def test_termination_wins_over_a_truncated_value():
    reward = T.tensor([[1.0, 2.0]])
    critic_value = T.tensor([[0.5, -0.5, 0.25]])
    terminated = T.tensor([[True, False]])
    truncated = T.tensor([[True, False]])

    _, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward,
        critic_value,
        terminated,
        truncated,
        gamma=GAMMA,
        gae_lambda=LAMBDA,
        truncated_value=T.tensor([[100.0, 100.0]]),
    )

    T.testing.assert_close(advantages, T.tensor([[0.5, 2.0 + 0.9 * 0.25 + 0.5]]))


def test_truncation_without_a_value_falls_back_to_the_stored_critic():
    reward = T.tensor([[1.0]])
    critic_value = T.tensor([[0.5, 4.0]])

    _, advantages = RolloutBuffer.compute_returns_and_advantages(
        reward,
        critic_value,
        T.tensor([[False]]),
        T.tensor([[True]]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    T.testing.assert_close(advantages, T.tensor([[1.0 + 0.9 * 4.0 - 0.5]]))
