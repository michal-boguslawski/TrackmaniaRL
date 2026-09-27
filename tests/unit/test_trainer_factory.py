"""Tests for implicit trainer registration and construction."""

import sys
from pkgutil import ModuleInfo

import pytest

from rl_lib.run_config import TrainerSettings
from rl_lib.training import factory
from rl_lib.training.ppo.trainer import PPOTrainer


def _isolate_registry(monkeypatch, registry: dict) -> None:
    """Replace the registry and stop discovery from repopulating it."""
    monkeypatch.setattr(factory, "_TRAINER_REGISTRY", registry)
    monkeypatch.setattr(factory, "_algorithms_discovered", True)


def test_ppo_registers_itself_when_its_package_is_imported():
    import rl_lib.training.ppo  # noqa: F401

    assert factory._TRAINER_REGISTRY["ppo"] is PPOTrainer


def test_create_trainer_builds_ppo_without_naming_it_in_the_factory(agent):
    trainer = factory.create_trainer("ppo", agent=agent, config=TrainerSettings())

    assert isinstance(trainer, PPOTrainer)


def test_create_trainer_constructs_a_registered_trainer(monkeypatch):
    class FakeTrainer:
        def __init__(self, label: str):
            self.label = label

    _isolate_registry(monkeypatch, {})
    factory.register_trainer("fake", FakeTrainer)

    trainer = factory.create_trainer("FAKE", label="selected")

    assert isinstance(trainer, FakeTrainer)
    assert trainer.label == "selected"


def test_create_trainer_rejects_an_unregistered_algorithm(monkeypatch):
    _isolate_registry(monkeypatch, {"ppo": object})

    with pytest.raises(ValueError, match="unknown training algorithm 'sac'"):
        factory.create_trainer("sac")


def test_register_trainer_rejects_a_duplicate_algorithm(monkeypatch):
    _isolate_registry(monkeypatch, {"ppo": object})

    with pytest.raises(ValueError, match="already registered"):
        factory.register_trainer("PPO", object)


def test_create_trainer_imports_algorithm_packages_it_does_not_know(tmp_path, monkeypatch):
    """A new algorithm package becomes selectable without editing the factory."""
    package = tmp_path / "discovered_algorithm"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from rl_lib.training.factory import register_trainer\n"
        "\n"
        "class DiscoveredTrainer:\n"
        "    pass\n"
        "\n"
        "register_trainer('discovered', DiscoveredTrainer)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(factory, "_TRAINER_REGISTRY", {})
    monkeypatch.setattr(factory, "_algorithms_discovered", False)
    monkeypatch.setattr(
        factory,
        "iter_modules",
        lambda path, prefix: [ModuleInfo(None, "discovered_algorithm", True)],
    )
    try:
        trainer = factory.create_trainer("discovered")
    finally:
        sys.modules.pop("discovered_algorithm", None)

    assert type(trainer).__name__ == "DiscoveredTrainer"
