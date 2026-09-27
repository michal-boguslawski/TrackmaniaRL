"""Unit tests for PPO optimization.

Covers minibatch assembly (the only place where the stored observation windows
are re-cut into per-sample blocks), the three loss terms, the update itself and
the KL/entropy bookkeeping around it.
"""

import math

import numpy as np
import pytest
import torch as T

from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.training.callbacks.base import Callback
from rl_lib.run_config import RolloutSettings, RunSettings, TrainerSettings
from rl_lib.training.ppo.losses import PPOLosses
from rl_lib.training.ppo.minibatches import get_iid_minibatches
from rl_lib.training.ppo.trainer import PPOTrainer


NUM_ENVS = 2
SIZE = 6
STACK_SIZE = 2
ACTION_DIM = 3
OBSERVATION_SHAPE = (96, 96, 1)
ENV_STRIDE = 50


class SpyCallback(Callback):
    def __init__(self):
        self.events: list[tuple] = []

    def on_start(self, *args, **kwargs):
        self.events.append(("start", kwargs.get("step"), sorted(kwargs.get("metrics", {}))))

    def on_minibatch(self, *args, **kwargs):
        self.events.append(("minibatch", kwargs.get("step")))

    def on_epoch(self, *args, **kwargs):
        self.events.append(("epoch",))

    def on_end(self, *args, **kwargs):
        self.events.append(("end", kwargs.get("step"), sorted(kwargs.get("metrics", {}))))


@pytest.fixture
def trainer_settings() -> TrainerSettings:
    return TrainerSettings()


@pytest.fixture
def trainer(agent: Agent, trainer_settings: TrainerSettings) -> PPOTrainer:
    return PPOTrainer(agent, trainer_settings)


@pytest.fixture
def ppo_losses(agent: Agent) -> PPOLosses:
    return PPOLosses(agent)


@pytest.fixture
def rollout(agent: Agent) -> dict[str, T.Tensor]:
    """A rollout collected through the agent, so the stored log-probs and values
    are the ones the update path has to reproduce."""

    buffer = RolloutBuffer(RolloutSettings(buffer_size=SIZE), stack_size=STACK_SIZE)
    done = T.zeros(NUM_ENVS, dtype=T.bool)
    for index in range(SIZE):
        state = T.randint(0, 256, (NUM_ENVS, *OBSERVATION_SHAPE), dtype=T.uint8)
        action, log_probs, value = agent.act(state, done)

        terminated = T.zeros(NUM_ENVS, dtype=T.bool)
        truncated = T.zeros(NUM_ENVS, dtype=T.bool)
        if index == 1:
            terminated[0] = True
            truncated[1] = True
        done = T.logical_or(terminated, truncated)

        truncated_value = T.zeros(NUM_ENVS)
        if truncated.any():
            final = T.randint(0, 256, (NUM_ENVS, *OBSERVATION_SHAPE), dtype=T.uint8)
            truncated_value[truncated] = agent.bootstrap_value(final)[truncated]

        buffer.add(
            RolloutStep(
                observation=state,
                action=action,
                critic_value=value,
                old_log_probs=log_probs,
                reward=T.rand(NUM_ENVS),
                terminated=terminated,
                truncated=truncated,
                truncated_value=truncated_value,
            )
        )
    return buffer.get()


@pytest.fixture
def tagged_batch() -> dict[str, T.Tensor]:
    """Synthetic rollout whose observations and transitions carry a recoverable
    `(env, step)` tag, so window alignment can be asserted exactly."""

    observation = T.zeros(NUM_ENVS, SIZE, *OBSERVATION_SHAPE, dtype=T.uint8)
    action = T.zeros(NUM_ENVS, SIZE - 1, ACTION_DIM)
    old_log_probs = T.zeros(NUM_ENVS, SIZE - 1, ACTION_DIM)
    dones = T.zeros(NUM_ENVS, SIZE, dtype=T.bool)
    for env in range(NUM_ENVS):
        for step in range(SIZE):
            # column 0 is the duplicated first step, column `step + 1` is state `step`
            observation[env, step, 0, 0, 0] = env * ENV_STRIDE + max(step - 1, 0)
        observation[env, 1:, 0, 0, 0] = env * ENV_STRIDE + T.arange(SIZE - 1)
        for step in range(SIZE - 1):
            action[env, step, 0] = env * ENV_STRIDE + step
            old_log_probs[env, step, 0] = env * ENV_STRIDE + step
    return {
        "observation": observation,
        "action": action,
        "old_log_probs": old_log_probs,
        "critic_value": T.zeros(NUM_ENVS, SIZE - 1),
        "returns": T.zeros(NUM_ENVS, SIZE - 1),
        "advantages": T.zeros(NUM_ENVS, SIZE - 1),
        "dones": dones,
    }


def test_minibatch_shapes_follow_the_rollout(trainer: PPOTrainer, rollout: dict[str, T.Tensor]):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))

    assert minibatch["observation"].shape == (4 * STACK_SIZE, *OBSERVATION_SHAPE)
    assert minibatch["dones"].shape == (4 * STACK_SIZE,)
    assert minibatch["action"].shape == (4, ACTION_DIM)
    assert minibatch["old_log_probs"].shape == (4, ACTION_DIM)
    assert minibatch["old_values"].shape == (4,)
    assert minibatch["returns"].shape == (4,)
    assert minibatch["advantages"].shape == (4,)


def test_minibatches_cover_every_transition_exactly_once(trainer: PPOTrainer, rollout):
    expected = sorted(rollout["advantages"].flatten().tolist())
    seen: list[float] = []
    for minibatch in get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True):
        seen.extend(minibatch["advantages"].flatten().tolist())

    assert len(seen) == len(expected)
    assert sorted(seen) == pytest.approx(expected, rel=1e-5)


def test_minibatch_count_and_tail_size(trainer: PPOTrainer, rollout):
    sizes = [mb["action"].shape[0] for mb in get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True)]

    assert sizes == [4, 4, 2]
    assert sum(sizes) == NUM_ENVS * (SIZE - 1)


def test_minibatches_cover_every_tagged_transition_once(trainer: PPOTrainer, tagged_batch):
    seen: list[int] = []
    for minibatch in get_iid_minibatches(tagged_batch, 4, STACK_SIZE, shuffle=True):
        seen.extend(int(value) for value in minibatch["action"][:, 0].tolist())

    assert sorted(seen) == sorted(
        env * ENV_STRIDE + step for env in range(NUM_ENVS) for step in range(SIZE - 1)
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_vectorized_minibatches_match_legacy_assembly(shuffle: bool):
    """Ensure vectorized indexing preserves the legacy minibatch contents."""
    num_envs = 3
    batch_size = 5
    stack_size = 3
    minibatch_size = 4
    observation_length = batch_size + stack_size - 1
    observation = T.arange(
        num_envs * observation_length * 2 * 2,
        dtype=T.uint8,
    ).reshape(num_envs, observation_length, 2, 2, 1)
    transition_indices = T.arange(num_envs * batch_size).reshape(num_envs, batch_size)
    batch = {
        "observation": observation,
        "action": (transition_indices.unsqueeze(-1) * 2 + T.arange(2)).float(),
        "old_log_probs": (transition_indices.unsqueeze(-1) * 3 + T.arange(3)).float(),
        "critic_value": transition_indices.float() + 0.25,
        "returns": transition_indices.float() + 10.5,
        "advantages": transition_indices.float() - 7.25,
        "dones": (T.arange(num_envs * observation_length).reshape(num_envs, -1) % 3) == 1,
    }

    def legacy_minibatches() -> list[dict[str, T.Tensor]]:
        """Assemble minibatches with the former per-sample indexing logic."""
        order = (
            np.random.permutation(batch_size * num_envs)
            if shuffle
            else np.arange(batch_size * num_envs)
        )
        result = []
        for start in range(0, batch_size * num_envs, minibatch_size):
            subindices = [
                (index % num_envs, index // num_envs)
                for index in order[start : start + minibatch_size]
            ]
            result.append(
                {
                    "observation": T.cat(
                        [
                            batch["observation"][env, time : time + stack_size]
                            for env, time in subindices
                        ],
                        dim=0,
                    ),
                    "action": T.stack(
                        [batch["action"][env, time] for env, time in subindices]
                    ),
                    "old_log_probs": T.stack(
                        [batch["old_log_probs"][env, time] for env, time in subindices]
                    ),
                    "old_values": T.stack(
                        [batch["critic_value"][env, time] for env, time in subindices]
                    ),
                    "returns": T.stack(
                        [batch["returns"][env, time] for env, time in subindices]
                    ),
                    "advantages": T.stack(
                        [batch["advantages"][env, time] for env, time in subindices]
                    ),
                    "dones": T.cat(
                        [
                            batch["dones"][env, time : time + stack_size]
                            for env, time in subindices
                        ],
                        dim=0,
                    ),
                }
            )
        return result

    rng_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected = legacy_minibatches()
        np.random.seed(1234)
        actual = list(
            get_iid_minibatches(
                batch, minibatch_size, stack_size, shuffle
            )
        )
    finally:
        np.random.set_state(rng_state)

    assert len(actual) == len(expected)
    for actual_minibatch, expected_minibatch in zip(actual, expected):
        assert actual_minibatch.keys() == expected_minibatch.keys()
        for key in actual_minibatch:
            T.testing.assert_close(
                actual_minibatch[key],
                expected_minibatch[key],
                rtol=0,
                atol=0,
                msg=lambda text, key=key: f"{key}: {text}",
            )


def test_minibatch_observation_windows_match_their_transition(trainer: PPOTrainer, tagged_batch):
    for minibatch in get_iid_minibatches(tagged_batch, 4, STACK_SIZE, shuffle=True):
        window = minibatch["observation"][:, 0, 0, 0]
        for index, tag in enumerate(minibatch["action"][:, 0].tolist()):
            env, step = divmod(int(tag), ENV_STRIDE)
            expected = T.tensor(
                [
                    env * ENV_STRIDE + max(step - 1, 0),
                    env * ENV_STRIDE + step,
                ],
                dtype=T.uint8,
            )
            T.testing.assert_close(
                window[index * STACK_SIZE : (index + 1) * STACK_SIZE], expected
            ), f"window for transition (env={env}, step={step})"


def test_minibatch_generation_leaves_the_batch_alone(rollout):
    before = {key: value.clone() for key, value in rollout.items()}

    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))

    assert minibatch["observation"].shape == (4 * STACK_SIZE, *OBSERVATION_SHAPE)
    for key, value in rollout.items():
        T.testing.assert_close(value, before[key], msg=lambda text, key=key: f"{key}: {text}")


def test_actor_loss_with_an_unchanged_policy_is_the_plain_surrogate(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    advantages = T.tensor([2.0, -1.0])
    log_probs = T.rand(2, ACTION_DIM)
    action_mean = T.zeros(2, ACTION_DIM)

    loss, metrics = ppo_losses.actor_loss(
        trainer.cfg, advantages, log_probs, log_probs, action_mean
    )

    T.testing.assert_close(loss, -advantages.mean())
    assert metrics["metrics/approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["metrics/clip_fraction"] == pytest.approx(0.0)
    assert metrics["metrics/ratio_max"] == pytest.approx(1.0, rel=1e-5)


def test_actor_loss_normalises_advantages_per_minibatch(
    agent: Agent, ppo_losses: PPOLosses
):
    trainer = PPOTrainer(agent, TrainerSettings(advantage_normalization_strategy="batch", mean_reg_coef=0.0))
    advantages = T.tensor([1.0, 2.0, 3.0])
    log_probs = T.rand(3, ACTION_DIM)
    action_mean = T.zeros(3, ACTION_DIM)

    loss, _ = ppo_losses.actor_loss(
        trainer.cfg, advantages, log_probs, log_probs, action_mean
    )

    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
    T.testing.assert_close(loss, -normalized.mean(), atol=1e-4, rtol=1e-4)


def test_actor_loss_clips_the_importance_ratio(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    trainer.cfg = trainer.cfg.model_copy(update={"ppo_epsilon": 0.2})
    advantages = T.ones(4)
    old_log_probs = T.zeros(4, ACTION_DIM)
    # one unit of log-ratio per action dim sums to log(1.65) -> above 1 + epsilon
    log_probs = T.full((4, ACTION_DIM), math.log(1.65) / ACTION_DIM)
    action_mean = T.zeros(4, ACTION_DIM)

    _, metrics = ppo_losses.actor_loss(
        trainer.cfg, advantages, log_probs, old_log_probs, action_mean
    )

    assert metrics["metrics/surrogate_loss"] == pytest.approx(-(1 + 0.2), rel=1e-4)
    assert metrics["metrics/clip_fraction"] == pytest.approx(1.0)
    assert metrics["metrics/ratio_max"] == pytest.approx(1.65, rel=1e-3)


def test_actor_loss_regularizes_toward_no_input_not_action_range_midpoint(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    advantages = T.zeros(2)
    log_probs = T.zeros(2, ACTION_DIM)
    # Zero is no input for all controls; gas/brake at 0.5 would apply both.
    action_mean = T.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    trainer.cfg = trainer.cfg.model_copy(update={"mean_reg_coef": 0.5})

    loss, metrics = ppo_losses.actor_loss(
        trainer.cfg, advantages, log_probs, log_probs, action_mean
    )

    assert metrics["metrics/mean_reg"] == pytest.approx(1.0 / 12.0)
    assert loss.item() == pytest.approx(1.0 / 24.0)


def test_actor_loss_metrics_are_reported_per_action_dim(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    _, metrics = ppo_losses.actor_loss(
        trainer.cfg,
        T.ones(3),
        T.rand(3, ACTION_DIM),
        T.rand(3, ACTION_DIM),
        T.rand(3, ACTION_DIM),
    )

    for key in (
        "metrics/actor_loss",
        "metrics/approx_kl",
        "metrics/clip_fraction",
        "metrics/surrogate_loss",
        "metrics/mean_reg",
        "metrics/mean_abs_max",
        "metrics/tanh_saturation_frac",
    ):
        assert key in metrics
    for index in range(ACTION_DIM):
        assert f"metrics/approx_kl_{index}" in metrics
        assert f"metrics/mean_abs_max_{index}" in metrics


def test_critic_loss_takes_the_worse_of_clipped_and_unclipped(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    trainer.cfg = trainer.cfg.model_copy(update={"ppo_epsilon": 0.2})
    returns = T.zeros(4)
    old_values = T.zeros(4)

    over, _ = ppo_losses.critic_loss(trainer.cfg, returns, T.ones(4), old_values)
    under, _ = ppo_losses.critic_loss(trainer.cfg, returns, -T.ones(4), old_values)

    # torch's Huber(delta=1) is 0.5 * x^2 inside the delta, so a residual of
    # 1.0 costs 0.5 while the clipped residual of 0.2 only costs 0.02: the
    # pessimistic maximum keeps the unclipped term
    assert over.item() == pytest.approx(0.5)
    assert under.item() == pytest.approx(0.5)


def test_critic_loss_ignores_the_clip_when_the_value_is_inside_it(
    trainer: PPOTrainer, ppo_losses: PPOLosses
):
    trainer.cfg = trainer.cfg.model_copy(update={"ppo_epsilon": 0.5})
    returns = T.zeros(2)

    loss, metrics = ppo_losses.critic_loss(
        trainer.cfg, returns, T.full((2,), 0.1), T.zeros(2)
    )

    # 0.1 is inside +-0.5, so the clipped value is the value itself
    assert loss.item() == pytest.approx(0.5 * 0.01)
    assert "loss/critic" in metrics


def test_entropy_loss_sums_the_beta_entropies(
    agent: Agent, ppo_losses: PPOLosses, observations
):
    obs = observations(4 * STACK_SIZE)
    action = T.rand(4, ACTION_DIM)
    dones = T.zeros(4 * STACK_SIZE, dtype=T.bool)
    _, _, dist, _ = agent.evaluate_actions(obs, action, dones)

    entropy_loss, metrics = ppo_losses.entropy_loss(dist)

    T.testing.assert_close(entropy_loss, dist.base_dist.entropy().sum(-1).mean())
    for index in range(ACTION_DIM):
        assert f"metrics/entropy_{index}" in metrics
        assert f"metrics/log_std_{index}" in metrics
    assert "loss/entropy" in metrics
    for index in range(ACTION_DIM):
        expected = dist.base_dist.variance[:, index].pow(0.5).log().mean()
        assert metrics[f"metrics/log_std_{index}"] == pytest.approx(expected.item(), rel=1e-4)
        assert metrics[f"metrics/entropy_{index}"] == pytest.approx(
            dist.base_dist.entropy()[:, index].mean().item(), rel=1e-4
        )


def test_entropy_loss_is_zero_for_a_uniform_beta(ppo_losses: PPOLosses):
    # the trainer reads `dist.base_dist`, i.e. the actor's transformed Beta
    dist = T.distributions.TransformedDistribution(
        T.distributions.Beta(T.ones(2, ACTION_DIM), T.ones(2, ACTION_DIM)),
        T.distributions.transforms.AffineTransform(loc=-1.0, scale=2.0),
    )

    entropy_loss, metrics = ppo_losses.entropy_loss(dist)

    T.testing.assert_close(entropy_loss, T.tensor(0.0))
    for index in range(ACTION_DIM):
        assert metrics[f"metrics/entropy_{index}"] == pytest.approx(0.0)
        # variance of Beta(1, 1) is 1/12
        assert metrics[f"metrics/log_std_{index}"] == pytest.approx(
            math.log(1 / 12) / 2, rel=1e-6
        )


def test_calculate_losses_combines_every_term(agent: Agent, trainer: PPOTrainer, rollout):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))

    (actor_loss, critic_loss, entropy_loss), metrics = trainer.calculate_losses(
        minibatch["advantages"],
        minibatch["returns"],
        minibatch["old_log_probs"],
        minibatch["old_values"],
        minibatch["observation"],
        minibatch["action"],
        minibatch["dones"],
    )

    for loss in (actor_loss, critic_loss, entropy_loss):
        assert loss.ndim == 0 and T.isfinite(loss)
    for key in ("loss/critic", "loss/entropy", "metrics/approx_kl", "metrics/actor_loss"):
        assert key in metrics


def test_train_step_updates_the_parameters(trainer: PPOTrainer, rollout):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    before = {name: param.clone() for name, param in trainer._agent._network.named_parameters()}

    metrics = trainer.train_step(**minibatch)

    assert any(not T.equal(param, before[name]) for name, param in trainer._agent._network.named_parameters())
    assert metrics["loss/total"] == pytest.approx(
        metrics["metrics/actor_loss"] + 0.5 * metrics["loss/critic"] - 0.001 * metrics["loss/entropy"],
        rel=1e-4,
    )
    assert "grad_norm/max" in metrics


def test_train_step_clips_gradients(trainer: PPOTrainer, rollout):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    trainer.train_step(**minibatch)

    norms = trainer._agent.get_partial_clip_grad_norms()

    for key in ("grad_norm/cnn", "grad_norm/sequence_encoder", "grad_norm/actor", "grad_norm/critic"):
        assert norms[key] <= 0.5 + 1e-4, f"{key} was not clipped to 0.5"


# ------------------------------------------------------------------ verbosity

CORE_LOSS_METRIC_KEYS = {
    "loss/critic",
    "loss/entropy",
    "metrics/actor_loss",
    "metrics/approx_kl",
}


def _uniform_beta_distribution():
    """A transformed uniform Beta, the shape the losses expect (they read
    ``base_dist``)."""
    return T.distributions.TransformedDistribution(
        T.distributions.Beta(T.ones(4, ACTION_DIM), T.ones(4, ACTION_DIM)),
        T.distributions.transforms.AffineTransform(loc=-1.0, scale=2.0),
    )


def _losses_at(agent: Agent, verbosity) -> PPOLosses:
    return PPOLosses(agent, verbosity)


def test_core_verbosity_still_computes_the_losses_and_the_kl(
    agent: Agent, trainer_settings: TrainerSettings, rollout
):
    """The update loop reads approx_kl to stop an epoch early, so verbosity can
    never drop it without changing how training behaves."""
    trainer = PPOTrainer(agent, trainer_settings, verbosity=1)
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))

    metrics = trainer.train_step(**minibatch)

    assert set(metrics) == CORE_LOSS_METRIC_KEYS


def test_core_verbosity_does_not_transfer_the_diagnostics_to_the_host(
    agent: Agent, trainer_settings: TrainerSettings, rollout
):
    trainer = PPOTrainer(agent, trainer_settings, verbosity=1)
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))

    metrics = trainer.train_step(**minibatch)

    for key in ("loss/total", "grad_norm/max", "metrics/clip_fraction", "metrics/log_std_0"):
        assert key not in metrics


def test_core_verbosity_keeps_the_losses_finite_and_updates_the_parameters(
    agent: Agent, trainer_settings: TrainerSettings, rollout
):
    trainer = PPOTrainer(agent, trainer_settings, verbosity=1)
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    before = {name: param.clone() for name, param in trainer._agent._network.named_parameters()}

    metrics = trainer.train_step(**minibatch)

    assert all(T.isfinite(T.tensor(value)) for value in metrics.values())
    assert any(
        not T.equal(param, before[name])
        for name, param in trainer._agent._network.named_parameters()
    )


@pytest.mark.parametrize("verbosity", [0, 1])
def test_low_verbosity_skips_the_rollout_batch_metrics(
    agent: Agent, trainer_settings: TrainerSettings, rollout, verbosity
):
    trainer = PPOTrainer(agent, trainer_settings, verbosity=verbosity)

    assert trainer._get_metrics_from_batch(rollout) == {}


def test_full_verbosity_keeps_the_rollout_batch_metrics(
    agent: Agent, trainer_settings: TrainerSettings, rollout
):
    trainer = PPOTrainer(agent, trainer_settings)

    metrics = trainer._get_metrics_from_batch(rollout)

    assert "rollout/explained_variance" in metrics
    assert "rollout/action_mean_0" in metrics


@pytest.mark.parametrize("verbosity", [0, 1])
def test_low_verbosity_losses_omit_the_per_action_diagnostics(
    agent: Agent, verbosity
):
    losses = _losses_at(agent, verbosity)
    dist = _uniform_beta_distribution()

    entropy_loss, entropy_metrics = losses.entropy_loss(dist)
    _, actor_metrics = losses.actor_loss(
        TrainerSettings(),
        T.ones(4),
        T.rand(4, ACTION_DIM),
        T.rand(4, ACTION_DIM),
        T.rand(4, ACTION_DIM),
    )

    assert set(entropy_metrics) == {"loss/entropy"}
    assert T.isfinite(entropy_loss)
    assert "metrics/entropy_0" not in entropy_metrics
    assert "metrics/log_std_0" not in entropy_metrics
    assert set(actor_metrics) == {"metrics/actor_loss", "metrics/approx_kl"}


def test_full_verbosity_losses_keep_the_per_action_diagnostics(agent: Agent):
    losses = _losses_at(agent, 2)

    _, entropy_metrics = losses.entropy_loss(_uniform_beta_distribution())

    for index in range(ACTION_DIM):
        assert f"metrics/entropy_{index}" in entropy_metrics
        assert f"metrics/log_std_{index}" in entropy_metrics


def test_trainer_rejects_an_unknown_verbosity(agent: Agent, trainer_settings: TrainerSettings):
    with pytest.raises(ValueError, match="verbosity must be one of"):
        PPOTrainer(agent, trainer_settings, verbosity=7)


def test_train_step_skips_a_non_finite_loss(trainer: PPOTrainer, rollout, monkeypatch):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    before = {name: param.clone() for name, param in trainer._agent._network.named_parameters()}
    monkeypatch.setattr(
        trainer,
        "calculate_losses",
        lambda *args, **kwargs: (
            (T.tensor(float("nan")), T.tensor(0.0), T.tensor(0.0)),
            {},
        ),
    )

    assert trainer.train_step(**minibatch) == {}
    for name, param in trainer._agent._network.named_parameters():
        T.testing.assert_close(param, before[name])


def test_train_runs_every_epoch_and_drives_the_callbacks(trainer: PPOTrainer, rollout):
    callback = SpyCallback()
    trainer._callbacks = type(trainer._callbacks)([callback])

    trainer.train(rollout, epochs=2, minibatch_size=4, training_step=0)

    kinds = [event[0] for event in callback.events]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    assert kinds.count("epoch") == 2, "an epoch boundary is reported per epoch"
    assert kinds.count("minibatch") == 2 * 3
    assert callback.events[0][1] == 0
    assert "rollout/returns" in callback.events[0][2]
    assert "training/entropy_coef" in callback.events[-1][2]


def test_train_normalises_advantages_globally(agent: Agent, rollout):
    trainer = PPOTrainer(agent, TrainerSettings(advantage_normalization_strategy="global"))
    before = rollout["advantages"].clone()

    trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)

    assert rollout["advantages"].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert rollout["advantages"].std().item() == pytest.approx(1.0, rel=1e-2)
    assert not T.allclose(rollout["advantages"], before)


def test_train_leaves_advantages_untouched_without_normalisation(trainer: PPOTrainer, rollout):
    before = rollout["advantages"].clone()

    trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)

    T.testing.assert_close(rollout["advantages"], before)


def test_train_stops_mid_epoch_when_the_kl_spikes(trainer: PPOTrainer, rollout, monkeypatch):
    trainer.cfg = trainer.cfg.model_copy(update={"hard_stop_kl": True})
    trainer.cfg = trainer.cfg.model_copy(update={"target_kl": 0.01})
    trainer.cfg = trainer.cfg.model_copy(update={"kl_warmup_steps": 0})
    monkeypatch.setattr(
        PPOTrainer,
        "train_step",
        lambda self, **kwargs: {"metrics/approx_kl": 10.0, "loss/total": 0.0},
    )
    callback = SpyCallback()
    trainer._callbacks = type(trainer._callbacks)([callback])

    trainer.train(rollout, epochs=3, minibatch_size=4, training_step=10)

    kinds = [event[0] for event in callback.events]
    # the run breaks out of the minibatch loop right after the first oversized
    # update, and that minibatch is the one that never reaches on_minibatch
    assert kinds.count("minibatch") == 0
    assert kinds.count("epoch") == 1, "the epoch is still closed out"


def test_train_keeps_going_while_the_kl_stays_below_target(trainer: PPOTrainer, rollout, monkeypatch):
    trainer.cfg = trainer.cfg.model_copy(update={"hard_stop_kl": True})
    trainer.cfg = trainer.cfg.model_copy(update={"target_kl": 0.01})
    trainer.cfg = trainer.cfg.model_copy(update={"kl_warmup_steps": 0})
    monkeypatch.setattr(
        PPOTrainer,
        "train_step",
        lambda self, **kwargs: {"metrics/approx_kl": 0.0, "loss/total": 0.0},
    )
    callback = SpyCallback()
    trainer._callbacks = type(trainer._callbacks)([callback])

    trainer.train(rollout, epochs=2, minibatch_size=4, training_step=10)

    kinds = [event[0] for event in callback.events]
    assert kinds.count("minibatch") == 6
    assert kinds.count("epoch") == 2


def test_train_respects_the_kl_warmup(trainer: PPOTrainer, rollout, monkeypatch):
    trainer.cfg = trainer.cfg.model_copy(update={"hard_stop_kl": True})
    trainer.cfg = trainer.cfg.model_copy(update={"target_kl": 0.01})
    trainer.cfg = trainer.cfg.model_copy(update={"kl_warmup_steps": 10_000})
    monkeypatch.setattr(
        PPOTrainer,
        "train_step",
        lambda self, **kwargs: {"metrics/approx_kl": 10.0, "loss/total": 0.0},
    )
    callback = SpyCallback()
    trainer._callbacks = type(trainer._callbacks)([callback])

    trainer.train(rollout, epochs=2, minibatch_size=4, training_step=1)

    kinds = [event[0] for event in callback.events]
    assert kinds.count("minibatch") == 6, "during the warmup the KL is only reported"


def test_train_reraises_update_failures(trainer: PPOTrainer, rollout, monkeypatch):
    def _boom(self, **kwargs):
        raise RuntimeError("update failed")

    monkeypatch.setattr(PPOTrainer, "train_step", _boom)

    with pytest.raises(RuntimeError, match="update failed"):
        trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)


def test_entropy_coefficient_decays_towards_a_floor(trainer: PPOTrainer, rollout):
    trainer._entropy_coef = 1e-2
    trainer.cfg = trainer.cfg.model_copy(update={"entropy_decay": 0.5})

    trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)
    assert trainer._entropy_coef == pytest.approx(5e-3)

    trainer._entropy_coef = 1e-5
    trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)
    assert trainer._entropy_coef == pytest.approx(1e-4)


def test_setup_train_builds_a_cosine_schedule(agent: Agent):
    trainer = PPOTrainer(agent, TrainerSettings())
    assert trainer._scheduler is None

    trainer.setup_train(RunSettings(total_steps=1000), RolloutSettings(buffer_size=128))

    assert type(trainer._scheduler).__name__ == "CosineAnnealingLR"
    assert trainer._scheduler.T_max == 1000 // 128 + 1
    assert trainer.config()["scheduler"] == "CosineAnnealingLR"


def test_train_steps_the_scheduler(agent: Agent, rollout):
    trainer = PPOTrainer(agent, TrainerSettings())
    trainer.setup_train(RunSettings(total_steps=10_000), RolloutSettings(buffer_size=SIZE))
    before = [group["lr"] for group in trainer._optimizer.param_groups]

    trainer.train(rollout, epochs=1, minibatch_size=4, training_step=0)

    after = [group["lr"] for group in trainer._optimizer.param_groups]
    assert all(new <= old for new, old in zip(after, before))


def test_optimizer_groups_use_the_configured_learning_rates(agent: Agent):
    trainer = PPOTrainer(agent, TrainerSettings(backbone_lr=1e-3, head_lr=1e-5, weight_decay=1e-2, optimizer_eps=1e-6))
    groups = trainer._optimizer.param_groups

    assert [group["lr"] for group in groups] == [1e-3, 1e-3, 1e-5, 1e-5]
    assert [group["weight_decay"] for group in groups] == [1e-2, 0.0, 1e-2, 0.0]
    assert trainer._optimizer.defaults["eps"] == 1e-6
    assert trainer.config()["optimizer_eps"] == 1e-6


def test_config_reports_the_hyperparameters(trainer: PPOTrainer):
    config = trainer.config()

    assert config["ppo_epsilon"] == trainer.cfg.ppo_epsilon
    assert config["critic_beta"] == trainer.cfg.critic_beta
    assert config["entropy_coef_init"] == trainer.cfg.entropy_coef
    assert config["advantage_normalization_strategy"] == "none"
    assert config["scheduler"] == "none"
    assert config["optimizer"] == "AdamW"


def test_metrics_from_batch_summarise_the_rollout(trainer: PPOTrainer, rollout):
    metrics = trainer._get_metrics_from_batch(rollout)

    for key in (
        "rollout/returns",
        "rollout/advantages_mean",
        "rollout/advantages_std",
        "rollout/critic_values",
        "rollout/old_log_probs",
        "rollout/explained_variance",
    ):
        assert key in metrics
    for index in range(ACTION_DIM):
        assert f"rollout/action_mean_{index}" in metrics
        assert f"rollout/action_std_{index}" in metrics

    assert metrics["rollout/returns"] == pytest.approx(rollout["returns"].mean().item(), rel=1e-5)
    assert metrics["rollout/advantages_mean"] == pytest.approx(rollout["advantages"].mean().item(), rel=1e-5)


def test_clip_grad_norm_returns_one_norm_per_parameter_group(trainer: PPOTrainer, rollout):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    log_probs, values, _, _ = trainer._agent.evaluate_actions(
        minibatch["observation"], minibatch["action"], minibatch["dones"]
    )
    (log_probs.sum() + values.sum()).backward()

    norms = trainer.clip_grad_norm(1.0, 1.0, 1.0)

    assert set(norms) == {
        "grad_norm/backbone_total",
        "grad_norm/actor_total",
        "grad_norm/critic_total",
    }
    assert all(isinstance(value, float) and value > 0 for value in norms.values())


def test_combined_gradient_norm_pass_matches_torch_group_clipping(
    trainer: PPOTrainer, rollout
):
    minibatch = next(get_iid_minibatches(rollout, 4, STACK_SIZE, shuffle=True))
    log_probs, values, _, _ = trainer._agent.evaluate_actions(
        minibatch["observation"], minibatch["action"], minibatch["dones"]
    )
    (log_probs.sum() + values.sum()).backward()

    modules = {
        "backbone": list(trainer._agent.network.cnn.parameters())
        + list(trainer._agent.network.sequence_encoder.parameters()),
        "actor": list(trainer._agent.network.actor.parameters()),
        "critic": list(trainer._agent.network.critic.parameters()),
    }
    limits = {"backbone": 0.2, "actor": 0.3, "critic": 0.4}
    reference_grads: dict[int, T.Tensor] = {}
    expected_norms: dict[str, float] = {}
    for group, parameters in modules.items():
        reference = [T.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
        for parameter, reference_parameter in zip(parameters, reference, strict=True):
            reference_parameter.grad = parameter.grad.detach().clone()
            reference_grads[id(parameter)] = reference_parameter.grad
        expected_norms[group] = T.nn.utils.clip_grad_norm_(
            reference, limits[group]
        ).item()

    actual_norms = trainer._agent.clip_grad_norms(
        limits["backbone"], limits["actor"], limits["critic"]
    )

    for group, parameters in modules.items():
        key = f"grad_norm/{group}_total"
        assert actual_norms[key].item() == pytest.approx(expected_norms[group], rel=1e-5)
        for parameter in parameters:
            T.testing.assert_close(parameter.grad, reference_grads[id(parameter)])


def test_act_delegates_to_the_agent(trainer: PPOTrainer, observations):
    obs = observations(NUM_ENVS)
    done = T.zeros(NUM_ENVS, dtype=T.bool)
    trainer._agent.reset()

    action, log_probs, value = trainer.act(obs, done)

    assert action.shape == (NUM_ENVS, ACTION_DIM)
    assert log_probs.shape == (NUM_ENVS, ACTION_DIM)
    assert value.shape == (NUM_ENVS,)


def test_bootstrap_value_delegates_to_the_agent(trainer: PPOTrainer, observations):
    obs = observations(NUM_ENVS)
    trainer._agent.reset()
    trainer._agent.act(obs, T.zeros(NUM_ENVS, dtype=T.bool))

    value = trainer.bootstrap_value(obs)

    T.testing.assert_close(value, trainer._agent.bootstrap_value(obs))


def test_step_env_delegates_to_the_agent(trainer: PPOTrainer, observations):
    state = observations(NUM_ENVS).numpy()
    next_state = observations(NUM_ENVS).numpy()
    done = T.zeros(NUM_ENVS, dtype=T.bool)
    stepped: list = []

    class _FakeEnv:
        def step(self, action):
            stepped.append(action)
            return (
                next_state,
                np.ones(NUM_ENVS, dtype=np.float32),
                np.zeros(NUM_ENVS, dtype=np.bool_),
                np.zeros(NUM_ENVS, dtype=np.bool_),
                {},
            )

    (
        returned_state, state_t, action, log_probs, value,
        reward, terminated, truncated, done_t, info,
    ) = trainer.step_env(_FakeEnv(), state, done, temperature=0.0)

    assert returned_state is next_state
    assert state_t.shape == (NUM_ENVS, *OBSERVATION_SHAPE)
    assert action.shape == (NUM_ENVS, ACTION_DIM)
    assert log_probs.shape == (NUM_ENVS, ACTION_DIM)
    assert value.shape == (NUM_ENVS,)
    assert reward.shape == (NUM_ENVS,)
    assert not terminated.any() and not truncated.any() and not done_t.any()
    assert info == {}
    assert len(stepped) == 1 and stepped[0].shape == (NUM_ENVS, ACTION_DIM)


def test_stack_size_and_device_are_the_agents(trainer: PPOTrainer, agent: Agent):
    assert trainer.stack_size == agent.stack_size
    assert T.device(trainer.device) == T.device(agent.device)


def test_repr_mentions_the_hyperparameters(trainer: PPOTrainer):
    text = repr(trainer)

    assert "PPOTrainer(" in text
    assert f"ppo_epsilon={trainer.cfg.ppo_epsilon}" in text
    assert f"target_kl={trainer.cfg.target_kl}" in text
    assert "scheduler=None" in text


def test_training_switches_the_agent_to_train_mode(agent: Agent):
    agent.eval()
    PPOTrainer(agent, TrainerSettings())

    assert agent.network.training


def test_network_config_is_the_agents(trainer: PPOTrainer, agent: Agent):
    assert trainer.network_config() == agent.network_config()
