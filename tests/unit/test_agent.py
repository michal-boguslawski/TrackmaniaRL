"""Unit tests for the observation-processing agent.

`Agent` owns everything between the vector environment and the network: uint8
NHWC -> normalised NCHW, the per-environment temporal window, and the mask that
invalidates frames recorded before an episode boundary. The mask and the
bootstrap path are the two pieces that silently corrupt PPO if they drift, so
they are pinned here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch as T

from rl_lib.agent import Agent
from rl_lib.networks.factory import Network
from rl_lib.run_config import AgentSettings
from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    CriticConfig,
    ConvLayerConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)


BATCH = 2
ACTION_DIM = 3
FEATURE_DIM = 4
OBSERVATION_SHAPE = (96, 96, 1)


@pytest.fixture
def big_stack_agent() -> Agent:
    config = NetworkConfig(
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
    network = Network(1, ACTION_DIM, 4, config)
    return Agent(network, device="cpu", config=AgentSettings(stack_size=4))


def test_preprocess_observation_normalises_and_moves_channels_first(agent: Agent, observations):
    obs = T.stack([observations(1, fill=0)[0], observations(1, fill=255)[0]])

    result = agent._preprocess_observation(obs)

    assert result.shape == (BATCH, 1, 96, 96)
    assert result.dtype is T.float32
    # uint8 0 and 255 map onto the ends of the [-1, 1] range
    T.testing.assert_close(result[0], T.full((1, 96, 96), -1.0), atol=1e-6, rtol=0)
    T.testing.assert_close(result[1], T.full((1, 96, 96), 1.0), atol=1e-6, rtol=0)

    noise = agent._preprocess_observation(observations(BATCH))
    assert (noise >= -1.0).all() and (noise <= 1.0).all()


def test_preprocess_observation_requires_uint8(agent: Agent):
    with pytest.raises(AssertionError):
        agent._preprocess_observation(T.zeros(BATCH, 1, 96, 96))


def test_preprocess_observation_normalisation_is_bitwise_identical_to_div_add(agent: Agent, observations):
    # The fused addcdiv kernel must round exactly like the previous
    # x / divisor + offset sequence; anything else silently perturbs every
    # observation the policy sees.
    obs = observations(BATCH)

    result = agent._preprocess_observation(obs)

    reference = obs.to(T.float32) / agent.cfg.observation_divisor + agent.cfg.observation_offset
    T.testing.assert_close(result, reference.permute(0, 3, 1, 2), atol=0, rtol=0)


def test_feature_extract_maps_observations_to_features(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)

    assert agent.feature_extract(obs).shape == (BATCH, FEATURE_DIM)


def test_features_window_pads_with_copies_of_the_first_frame(agent: Agent):
    features = T.rand(BATCH, FEATURE_DIM)

    window = agent._get_features_window(features)

    assert window.shape == (BATCH, 2, FEATURE_DIM)
    T.testing.assert_close(window[:, 0], features)
    T.testing.assert_close(window[:, 1], features)
    # the padding must be a copy, not the same storage
    assert window[:, 0].data_ptr() != features.data_ptr()


def test_features_window_slides_once_full(agent: Agent):
    first = T.rand(BATCH, FEATURE_DIM)
    second = T.rand(BATCH, FEATURE_DIM)

    agent._get_features_window(first)
    window = agent._get_features_window(second)

    T.testing.assert_close(window[:, 0], first)
    T.testing.assert_close(window[:, 1], second)


def test_mask_window_is_empty_at_the_start_of_training(agent: Agent):
    mask = agent._get_mask_window(T.zeros(BATCH, dtype=T.bool))

    assert mask.shape == (BATCH, 2)
    assert not mask.any()


def test_mask_window_marks_frames_older_than_the_latest_done(big_stack_agent: Agent):
    """A frame recorded before a done belongs to a previous episode and must be
    zeroed; the frame of the current step is always kept."""

    agent = big_stack_agent
    masks = []
    for done in (
        T.tensor([False, False, False]),
        T.tensor([False, True, False]),
        T.tensor([True, False, False]),
        T.tensor([False, True, True]),
    ):
        masks.append(agent._get_mask_window(done).clone())

    assert masks[0].sum() == 0
    T.testing.assert_close(
        masks[1],
        T.tensor(
            [
                [False, False, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ),
    )
    # env 0 just finished an episode, env 1 finished one two steps ago: its
    # oldest frame is the only one older than the latest done.
    T.testing.assert_close(
        masks[2],
        T.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
                [False, False, False, False],
            ]
        ),
    )
    T.testing.assert_close(
        masks[3],
        T.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
                [True, True, True, False],
            ]
        ),
    )


def test_mask_window_always_keeps_the_current_frame(agent: Agent):
    for value in (False, True):
        agent.reset()
        mask = agent._get_mask_window(T.full((BATCH,), value, dtype=T.bool))
        assert not mask[:, -1].any(), "the newest frame is never stale"


def test_temporal_encode_shape_without_mask(agent: Agent):
    encoded = agent.temporal_encode(T.rand(BATCH, 2, FEATURE_DIM))

    assert encoded.shape == (BATCH, FEATURE_DIM)


def test_temporal_encode_ignores_masked_frames(agent: Agent):
    features = T.rand(BATCH, 2, FEATURE_DIM)
    mask = T.tensor([[True, False], [False, True]])

    encoded = agent.temporal_encode(features, mask)
    zeroed = agent.temporal_encode(features.masked_fill(mask.unsqueeze(-1), 0.0))

    T.testing.assert_close(encoded, zeroed)
    assert not T.allclose(encoded[0], encoded[1], atol=1e-6)


def test_temporal_encode_rejects_a_mismatched_mask(agent: Agent):
    with pytest.raises(AssertionError):
        agent.temporal_encode(T.rand(BATCH, 2, FEATURE_DIM), T.zeros(BATCH, 3, dtype=T.bool))


def test_act_returns_action_log_probs_and_value(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)
    done = T.zeros(BATCH, dtype=T.bool)

    action, log_probs, value = agent.act(obs, done)

    assert action.shape == (BATCH, ACTION_DIM)
    assert log_probs.shape == (BATCH, ACTION_DIM)
    assert value.shape == (BATCH,)
    assert (action > -1.0).all() and (action < 1.0).all()
    assert T.isfinite(log_probs).all()
    # collection runs under no_grad
    assert not action.requires_grad and not log_probs.requires_grad and not value.requires_grad


def test_act_is_deterministic_at_zero_temperature(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)
    done = T.zeros(BATCH, dtype=T.bool)

    agent.reset()
    greedy, _, _ = agent.act(obs, done, temperature=0.0)
    agent.reset()
    greedy_again, _, _ = agent.act(obs, done, temperature=0.0)
    agent.reset()
    sampled, _, _ = agent.act(obs, done, temperature=1.0)

    T.testing.assert_close(greedy, greedy_again, atol=0, rtol=0)
    assert not T.allclose(greedy, sampled, atol=1e-3)


def test_act_at_zero_temperature_returns_the_policy_mean(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)
    done = T.zeros(BATCH, dtype=T.bool)

    action, _, _ = agent.act(obs, done, temperature=0.0)

    agent.reset()
    features = agent.feature_extract(obs)
    _, expected_mean, _ = agent.heads(agent.temporal_encode(agent._get_features_window(features)))

    T.testing.assert_close(action, expected_mean)


def test_act_raises_when_log_probs_are_not_finite(agent: Agent, monkeypatch):
    class _NaNDistribution:
        base_dist = SimpleNamespace(
            concentration0=T.zeros(1), concentration1=T.zeros(1)
        )

        def sample(self):
            return T.zeros(BATCH, ACTION_DIM)

        def log_prob(self, action):
            return T.full((BATCH, ACTION_DIM), float("nan"))

    monkeypatch.setattr(
        agent, "heads", lambda *args, **kwargs: (_NaNDistribution(), T.zeros(BATCH, ACTION_DIM), T.zeros(BATCH))
    )

    with pytest.raises(ValueError, match="log_probs are not finite"):
        agent.act(
            T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8),
            T.zeros(BATCH, dtype=T.bool),
        )


def test_heads_delegates_to_the_network(agent: Agent):
    temporal = T.rand(BATCH, FEATURE_DIM)

    agent_dist, agent_mean, agent_value = agent.heads(temporal, 0.7)
    net_dist, net_mean, net_value = agent._network.heads(temporal, 0.7)

    T.testing.assert_close(agent_mean, net_mean)
    T.testing.assert_close(agent_value, net_value)

    probe = T.rand(BATCH, ACTION_DIM)
    T.testing.assert_close(agent_dist.log_prob(probe), net_dist.log_prob(probe))


def test_bootstrap_value_does_not_advance_the_policy_history(agent: Agent, observations):
    obs = observations(BATCH)
    done = T.zeros(BATCH, dtype=T.bool)
    agent.act(obs, done)

    obs_window_before = [frame.clone() for frame in agent._obs_window]
    done_window_before = [frame.clone() for frame in agent._done_window]

    value = agent.bootstrap_value(obs)

    assert value.shape == (BATCH,)
    assert len(agent._obs_window) == len(obs_window_before)
    assert len(agent._done_window) == len(done_window_before)
    for before, after in zip(obs_window_before, agent._obs_window):
        T.testing.assert_close(before, after)
    for before, after in zip(done_window_before, agent._done_window):
        assert T.equal(before, after)


def test_bootstrap_value_appends_the_successor_with_a_false_done(agent: Agent, observations):
    obs = observations(BATCH)
    done = T.zeros(BATCH, dtype=T.bool)
    agent.act(obs, done)
    successor = observations(BATCH)

    value = agent.bootstrap_value(successor)

    prior_features = list(agent._obs_window)[-1:]
    window = T.stack([*prior_features, agent.feature_extract(successor)], dim=1)
    prior_dones = list(agent._done_window)[-1:]
    dones = T.stack([*prior_dones, T.zeros(BATCH, dtype=T.bool)], dim=1)
    mask = dones.logical_not() & (dones.flip(1).cumsum(1).flip(1) > 0)

    T.testing.assert_close(value, agent._network.critic(agent.temporal_encode(window, mask)))


def test_bootstrap_value_is_deterministic(agent: Agent, observations):
    obs = observations(BATCH)
    agent.act(obs, T.zeros(BATCH, dtype=T.bool))
    successor = observations(BATCH)

    T.testing.assert_close(
        agent.bootstrap_value(successor), agent.bootstrap_value(successor), atol=0, rtol=0
    )


def test_evaluate_actions_reproduces_the_log_probs_from_act(agent: Agent, observations):
    """PPO's importance ratio is only 1 if the update path sees exactly the
    distribution the rollout was collected with."""

    stack_size = agent.stack_size
    first_obs, second_obs = observations(BATCH), observations(BATCH)
    first_done = T.tensor([False, True])
    second_done = T.tensor([True, False])

    first_action, first_log_probs, first_value = agent.act(first_obs, first_done)
    action, log_probs, value = agent.act(second_obs, second_done)

    # the minibatch layout produced by the PPO minibatch generator:
    # each sample owns `stack_size` consecutive observation rows
    obs_window = T.stack([first_obs, second_obs], dim=1).reshape(-1, *OBSERVATION_SHAPE)
    done_window = T.stack([first_done, second_done], dim=1).reshape(-1)

    re_log_probs, re_values, dist, action_mean = agent.evaluate_actions(
        obs_window, action, done_window
    )

    assert re_log_probs.shape == (BATCH, ACTION_DIM)
    assert re_values.shape == (BATCH,)
    assert action_mean.shape == (BATCH, ACTION_DIM)
    T.testing.assert_close(re_log_probs, log_probs, atol=1e-4, rtol=1e-4)
    T.testing.assert_close(re_values, value, atol=1e-4, rtol=1e-4)
    assert re_log_probs.requires_grad, "the update path must stay differentiable"
    assert first_action.shape == (BATCH, ACTION_DIM)
    assert first_log_probs.shape == (BATCH, ACTION_DIM)
    assert first_value.shape == (BATCH,)


def test_evaluate_actions_keeps_gradients(agent: Agent, observations):
    obs_window = observations(BATCH * agent.stack_size)
    done_window = T.zeros(BATCH * agent.stack_size, dtype=T.bool)

    log_probs, values, dist, action_mean = agent.evaluate_actions(
        obs_window, T.rand(BATCH, ACTION_DIM), done_window
    )

    (log_probs.sum() + values.sum() + action_mean.pow(2).mean()).backward()
    for name, param in agent._network.named_parameters():
        assert param.grad is not None, f"{name} has no gradient from evaluate_actions"
        assert T.isfinite(param.grad).all()


def test_network_parameter_groups_partition_the_parameters(agent: Agent):
    groups = agent.network_parameter_groups()

    assert set(groups) == {"backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}

    by_id = {id(p): group for group, params in groups.items() for p in params}
    assert len(by_id) == len(list(agent._network.parameters())), "every parameter is grouped once"
    for params in groups.values():
        assert params, "no parameter group may be empty, the optimizer would reject it"


def test_network_parameter_groups_exclude_biases_and_norms_from_decay(agent: Agent):
    groups = agent.network_parameter_groups()

    def names(group: str) -> set[str]:
        return {id(p) for p in groups[group]}

    backbone_decay = names("backbone_decay")
    backbone_no_decay = names("backbone_no_decay")
    assert not backbone_decay & backbone_no_decay

    for module, prefix in ((agent._network.cnn, "cnn"), (agent._network.sequence_encoder, "temporal")):
        for name, param in module.named_parameters():
            should_decay = "bias" not in name and "norm" not in name.lower()
            grouped = (id(param) in backbone_decay) if should_decay else (id(param) in backbone_no_decay)
            assert grouped, f"{prefix}.{name} landed in the wrong weight-decay group"

    head_no_decay = {id(p) for p in groups["head_no_decay"]}
    head_decay = {id(p) for p in groups["head_decay"]}
    assert not head_decay & head_no_decay
    for module in (agent._network.actor, agent._network.critic):
        for name, param in module.named_parameters():
            expected = head_no_decay if "bias" in name else head_decay
            assert id(param) in expected, f"{name} landed in the wrong head group"


def test_clip_grad_norm_scales_gradients_down(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)
    log_probs, values, _, action_mean = agent.evaluate_actions(
        obs, T.rand(BATCH, ACTION_DIM), T.zeros(BATCH, dtype=T.bool)
    )
    (-log_probs.sum() + values.sum() + action_mean.pow(2).mean()).backward()

    before = agent.clip_grad_norm(1e-4)
    after = agent.clip_grad_norm(1e-4)

    assert isinstance(before, T.Tensor)
    assert after.item() <= 1e-4 + 1e-6


def test_partial_grad_norms_cover_every_submodule(agent: Agent):
    obs = T.randint(0, 256, (BATCH, *OBSERVATION_SHAPE), dtype=T.uint8)
    log_probs, _, _, _ = agent.evaluate_actions(
        obs, T.rand(BATCH, ACTION_DIM), T.zeros(BATCH, dtype=T.bool)
    )
    log_probs.sum().backward()

    norms = agent.get_partial_clip_grad_norms()

    assert set(norms) == {
        "grad_norm/cnn",
        "grad_norm/sequence_encoder",
        "grad_norm/actor",
        "grad_norm/critic",
        "grad_norm/max",
    }
    assert all(isinstance(value, float) and value >= 0.0 for value in norms.values())


def test_save_and_load_state_dict_roundtrip(tmp_path, agent: Agent):
    path = str(tmp_path / "agent.pt")
    agent.save_state_dict(path)
    expected = {k: v.clone() for k, v in agent._network.state_dict().items()}

    with T.no_grad():
        for param in agent._network.parameters():
            param.add_(1.0)
    agent.load_state_dict(path)

    for key, value in agent._network.state_dict().items():
        T.testing.assert_close(value, expected[key])


def test_reset_clears_the_temporal_windows(agent: Agent, observations):
    obs = observations(BATCH)
    agent.act(obs, T.zeros(BATCH, dtype=T.bool))
    assert len(agent._obs_window) == agent.stack_size

    agent.reset()

    assert len(agent._obs_window) == 0
    assert len(agent._done_window) == 0


def test_step_env_owns_the_numpy_conversion_and_done_logic(agent: Agent, observations):
    state = observations(BATCH).numpy()
    next_state = observations(BATCH).numpy()
    done = T.zeros(BATCH, dtype=T.bool)
    reward = np.array([1.0, 2.0], dtype=np.float32)
    terminated = np.array([True, False])
    truncated = np.array([False, True])

    class _FakeEnv:
        def step(self, action):
            self.action = action
            return next_state, reward, terminated, truncated, {"info": 1}

    env = _FakeEnv()

    (
        returned_next_state,
        returned_state,
        action,
        log_probs,
        value,
        returned_reward,
        returned_terminated,
        returned_truncated,
        new_done,
        info,
    ) = agent.step_env(env, state, done)

    assert isinstance(returned_next_state, np.ndarray)
    T.testing.assert_close(returned_state, T.from_numpy(state))
    assert action.shape == (BATCH, ACTION_DIM)
    assert log_probs.shape == (BATCH, ACTION_DIM)
    assert value.shape == (BATCH,)
    assert list(returned_reward) == [1.0, 2.0]
    T.testing.assert_close(returned_terminated, T.tensor([True, False]))
    T.testing.assert_close(returned_truncated, T.tensor([False, True]))
    T.testing.assert_close(new_done, T.tensor([True, True]))
    assert agent.last_step_had_truncation
    assert info == {"info": 1}
    # the numpy action handed to the environment matches the sampled one
    assert env.action.shape == (BATCH, ACTION_DIM)
    T.testing.assert_close(T.from_numpy(env.action), action.cpu())


def test_network_config_matches_the_network(agent: Agent):
    assert agent.network_config() == agent._network.config()
    assert agent.network_config()["total_params"] == sum(
        p.numel() for p in agent._network.parameters()
    )


def test_eval_and_train_toggle_the_network(agent: Agent):
    agent.eval()
    assert not agent._network.training

    agent.train()
    assert agent._network.training


def test_device_and_stack_size_properties(agent: Agent):
    assert agent.stack_size == 2
    assert T.device(agent.device).type == "cpu"


def test_action_transform_delegates_to_the_network(agent: Agent):
    action = agent.action_transform(T.rand(BATCH, ACTION_DIM))

    assert action.shape == (BATCH, ACTION_DIM)
