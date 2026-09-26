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
