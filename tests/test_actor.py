import pytest
import torch as T
from torch.distributions import Distribution

from rl_lib.networks.actor import Actor


@pytest.fixture
def actor() -> Actor:
    return Actor(3)

def test_actor_forward(actor: Actor):
    x = T.rand(10, 256)
    dist = actor(x)
    assert isinstance(dist, Distribution)

    actions = dist.sample()
    assert actions.shape == (10, 3)

def test_sampled_actions_within_bounds(actor: Actor):
    x = T.randn(1000, 256)
    action = actor(x).sample()
    assert (action[:, 0] > -1).all() and (action[:, 0] < 1).all()
    assert (action[:, 1] > 0).all() and (action[:, 1] < 1).all()
    assert (action[:, 2] > 0).all() and (action[:, 2] < 1).all()

def test_temperature_affects_std(actor: Actor):
    x = T.randn(10, 256)
    std_low = actor(x, temperature=0.1).base_dist.scale
    std_high = actor(x, temperature=2.0).base_dist.scale
    assert (std_low < std_high).all()

def test_temperature_zero_is_degenerate_but_finite(actor: Actor):
    x = T.randn(10, 256)
    dist = actor(x, temperature=1e-6)
    assert T.isfinite(dist.base_dist.scale).all()  # std=0 is valid for Normal, just check it doesn't NaN downstream

def test_log_std_clamped_range(actor: Actor):
    with T.no_grad():
        actor._log_std.fill_(100.)  # force out-of-range
    x = T.randn(10, 256)
    std = actor(x).base_dist.scale
    assert T.allclose(std, T.full_like(std, T.tensor(0.5).exp()), atol=1e-4)

    with T.no_grad():
        actor._log_std.fill_(-100.)
    std = actor(x).base_dist.scale
    assert T.allclose(std, T.full_like(std, T.tensor(-2.0).exp()), atol=1e-4)

def test_gradients_flow_to_all_parameters(actor: Actor):
    x = T.randn(10, 256)
    dist = actor(x)
    loss = dist.log_prob(dist.rsample()).sum()  # rsample needed for reparameterized grad through mean/std
    loss.backward()
    for name, p in actor.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert T.isfinite(p.grad).all(), f"{name} has non-finite gradient"

def test_log_prob_finite_near_tanh_boundary(actor: Actor):
    x = T.randn(10, 256) * 10  # push mean far from 0 to stress-test tanh saturation
    dist = actor(x)
    action = dist.sample()
    log_prob = dist.log_prob(action.clamp(-0.999, 0.999))  # simulate near-boundary samples
    assert T.isfinite(log_prob).all()

def test_single_sample_forward(actor: Actor):
    x = T.randn(1, 256)
    dist = actor(x)
    assert dist.sample().shape == (1, 3)
