"""Unit tests for vector-environment construction."""

from importlib import import_module

from gymnasium.vector import AutoresetMode


def test_make_env_uses_same_step_autoreset(monkeypatch):
    make_env_module = import_module("rl_lib.envs.make_env")
    captured_kwargs = {}

    def fake_make_vec(env_id, **kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(make_env_module, "make_vec", fake_make_vec)

    make_env_module.make_env("test-env", num_envs=2)

    assert captured_kwargs["vector_kwargs"]["autoreset_mode"] is AutoresetMode.SAME_STEP


def test_make_env_uses_configured_reward_normalization(monkeypatch):
    make_env_module = import_module("rl_lib.envs.make_env")
    captured = {}
    sentinel = object()

    def fake_normalize(env, gamma, epsilon):
        captured.update(gamma=gamma, epsilon=epsilon)
        return env

    monkeypatch.setattr(make_env_module, "make_vec", lambda env_id, **kwargs: sentinel)
    monkeypatch.setattr(make_env_module, "NormalizeReward", fake_normalize)

    assert make_env_module.make_env(
        "test-env",
        num_envs=1,
        normalize_rewards=True,
        reward_normalization_gamma=0.8,
        reward_normalization_epsilon=1e-6,
    ) is sentinel
    assert captured == {"gamma": 0.8, "epsilon": 1e-6}
