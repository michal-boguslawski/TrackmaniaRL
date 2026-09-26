"""Unit tests for the Beta-distribution policy head.

The actor emits a `TransformedDistribution` over a `Beta` base: the Beta is
mapped by an affine transform onto the environment's action bounds
(steering in [-1, 1], gas/brake in [0, 1]). `temperature` divides both
concentrations, which shrinks the variance and - because the concentrations
carry a `+ 1` offset - also pulls the mean towards the centre of the action box.
"""

import pytest
import torch as T
from torch.distributions import Distribution, TransformedDistribution

from rl_lib.networks.actor import Actor, StableTanhTransform
from rl_lib.networks.config import ActorConfig


ACTION_DIM = 3
IN_DIM = 8
BATCH = 10

# steering, gas, brake
ACTION_LOW = T.tensor([-1.0, 0.0, 0.0])
ACTION_HIGH = T.tensor([1.0, 1.0, 1.0])
# centre of the transformed support == the neutral action
NEUTRAL_MEAN = T.tensor([0.0, 0.5, 0.5])


@pytest.fixture
def actor() -> Actor:
    return Actor(action_dim=ACTION_DIM, in_dim=IN_DIM, hidden_dim=8)


def test_forward_returns_distribution_and_transformed_mean(actor: Actor):
    dist, action_mean = actor(T.randn(BATCH, IN_DIM))

    assert isinstance(dist, TransformedDistribution)
    assert isinstance(dist, Distribution)
    assert dist.sample().shape == (BATCH, ACTION_DIM)
    assert action_mean.shape == (BATCH, ACTION_DIM)
    assert T.isfinite(action_mean).all()

    # the head is elementwise, so log-probs keep one entry per action dim
    assert dist.log_prob(dist.sample()).shape == (BATCH, ACTION_DIM)


def test_sampled_actions_respect_environment_bounds(actor: Actor):
    dist, _ = actor(T.randn(2000, IN_DIM))
    action = dist.sample()

    assert (action > ACTION_LOW).all(), "steering/gas/brake must stay strictly inside the support"
    assert (action < ACTION_HIGH).all()


def test_forward_mean_is_in_transformed_action_coordinates():
    """The regularisation term in the PPO actor loss is applied to the mean the
    head returns, so it has to live in action space: a neutral policy encodes
    steering 0, not the left edge of the Beta support."""

    actor = Actor(ACTION_DIM, IN_DIM, hidden_dim=8)
    with T.no_grad():
        actor._network[-1].weight.zero_()
        actor._network[-1].bias.zero_()

    _, action_mean = actor(T.zeros(1, IN_DIM))

    T.testing.assert_close(action_mean, NEUTRAL_MEAN.view(1, ACTION_DIM), atol=1e-6, rtol=0)


def test_deterministic_head_is_the_untempered_policy(actor: Actor):
    """`forward_deterministic` is what `heads` uses at temperature 0, so it has
    to agree with the untempered head instead of some other scaling."""

    # asymmetric concentrations, otherwise every temperature sits on the
    # neutral action and the comparison would be vacuous
    with T.no_grad():
        actor._network[-1].weight.zero_()
        actor._network[-1].bias.copy_(T.tensor([1.0, -1.0, 2.0, -1.0, 1.0, -1.0]))
    x = T.randn(BATCH, IN_DIM)

    _, deterministic_mean = actor.forward_deterministic(x)
    _, untempered_mean = actor(x, temperature=1.0)
    _, tempered_mean = actor(x, temperature=0.5)

    T.testing.assert_close(deterministic_mean, untempered_mean, atol=0, rtol=0)
    assert not T.allclose(deterministic_mean, tempered_mean, atol=1e-2)


def test_lower_temperature_narrows_the_distribution(actor: Actor):
    x = T.randn(BATCH, IN_DIM)

    cold, _ = actor(x, temperature=0.1)
    warm, _ = actor(x, temperature=2.0)

    assert (cold.base_dist.concentration0 > warm.base_dist.concentration0).all()
    assert (cold.base_dist.concentration1 > warm.base_dist.concentration1).all()
    assert (cold.base_dist.variance < warm.base_dist.variance).all()


def test_higher_temperature_pulls_the_mean_to_the_centre_of_the_action_box(actor: Actor):
    x = T.randn(BATCH, IN_DIM)

    _, low_temperature_mean = actor(x, temperature=1.0)
    _, high_temperature_mean = actor(x, temperature=1e3)

    T.testing.assert_close(high_temperature_mean, NEUTRAL_MEAN.expand(BATCH, ACTION_DIM), atol=1e-3, rtol=0)
    assert (low_temperature_mean[:, 1:] - 0.5).abs().sum() > 0.0


def test_near_zero_temperature_stays_finite(actor: Actor):
    x = T.randn(BATCH, IN_DIM) * 10

    dist, action_mean = actor(x, temperature=1e-6)

    assert T.isfinite(dist.base_dist.concentration0).all()
    assert T.isfinite(dist.base_dist.concentration1).all()
    assert T.isfinite(action_mean).all()
    assert T.isfinite(dist.log_prob(dist.rsample())).all()
    assert (dist.base_dist.variance < actor(x, temperature=1.0)[0].base_dist.variance).all()


def test_concentrations_are_always_above_one(actor: Actor):
    """softplus(raw) / temperature + 1 keeps the Beta parameters valid even for
    very negative pre-activations and extreme temperatures."""

    x = T.randn(BATCH, IN_DIM) * 10

    for temperature in (1e-3, 1.0, 1e3):
        dist, _ = actor(x, temperature=temperature)
        assert (dist.base_dist.concentration0 > 1).all()
        assert (dist.base_dist.concentration1 > 1).all()


def test_gradients_flow_to_all_parameters(actor: Actor):
    x = T.randn(BATCH, IN_DIM)
    dist, _ = actor(x)

    (-dist.log_prob(dist.rsample())).sum().backward()

    for name, param in actor.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert T.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_gradients_flow_through_the_returned_mean(actor: Actor):
    """`mean_reg` in the PPO actor loss backpropagates through this tensor."""

    _, action_mean = actor(T.randn(BATCH, IN_DIM))
    action_mean.pow(2).mean().backward()

    for name, param in actor.named_parameters():
        assert param.grad is not None, f"{name} has no gradient through the mean"
        assert param.grad.abs().sum() > 0, f"{name} received a zero gradient through the mean"


def test_log_prob_is_finite_near_the_support_boundary(actor: Actor):
    dist, _ = actor(T.randn(BATCH, IN_DIM) * 10)

    action = dist.sample()
    edge = (action - ACTION_LOW).clamp_min(1e-6) + ACTION_LOW
    edge = T.minimum(edge, ACTION_HIGH - 1e-6)

    assert T.isfinite(dist.log_prob(action)).all()
    assert T.isfinite(dist.log_prob(edge)).all()


def test_single_sample_forward(actor: Actor):
    dist, action_mean = actor(T.randn(1, IN_DIM))

    assert dist.sample().shape == (1, ACTION_DIM)
    assert action_mean.shape == (1, ACTION_DIM)


def test_config_reports_dims_and_param_counts(actor: Actor):
    config = actor.config()
    total = sum(p.numel() for p in actor.parameters())

    assert config["action_dim"] == ACTION_DIM
    assert config["in_dim"] == IN_DIM
    assert config["hidden_layers"][0]["out_dim"] == 8
    assert config["n_params"] == total
    assert config["n_trainable_params"] == total


def test_repr_mentions_its_dimensions(actor: Actor):
    text = repr(actor)

    assert "Actor(" in text
    assert f"action_dim={ACTION_DIM}" in text
    assert f"in_dim={IN_DIM}" in text


def test_out_dim(actor: Actor):
    assert actor.out_dim == ACTION_DIM


def test_affine_buffers_are_registered(actor: Actor):
    buffers = dict(actor.named_buffers())

    assert set(buffers) == {"_affine_loc", "_affine_scale"}
    T.testing.assert_close(buffers["_affine_loc"], T.tensor([-1.0, 0.0, 0.0]))
    T.testing.assert_close(buffers["_affine_scale"], T.tensor([2.0, 1.0, 1.0]))


def test_actor_exposes_only_the_mlp_parameters(actor: Actor):
    names = {name for name, _ in actor.named_parameters()}

    assert names == {
        "_network.0.weight",
        "_network.0.bias",
        "_network.2.weight",
        "_network.2.bias",
    }
    assert isinstance(actor.cfg, ActorConfig)


@pytest.mark.parametrize("y", [-1.0, -0.999, 0.0, 0.5, 0.999, 1.0])
def test_stable_tanh_log_abs_det_jacobian_is_finite(y: float):
    transform = StableTanhTransform(epsilon=1e-6)
    y = T.tensor([[y]])

    log_det = transform.log_abs_det_jacobian(T.zeros_like(y), y)

    assert T.isfinite(log_det).all(), "log(1 - tanh(x)^2) must not blow up at saturation"


def test_stable_tanh_inverse_inverts_forward_in_the_interior():
    transform = StableTanhTransform(epsilon=1e-6)
    x = T.tensor([[-2.0], [0.0], [2.0]])

    recovered = transform._inverse(transform(x))

    T.testing.assert_close(recovered, x, atol=1e-3, rtol=1e-3)


def test_stable_tanh_inverse_clamps_instead_of_logging_zero():
    transform = StableTanhTransform(epsilon=1e-6)

    recovered = transform._inverse(T.tensor([[-1.0], [1.0]]))

    assert T.isfinite(recovered).all()
    assert recovered.abs().max() < 10.0
