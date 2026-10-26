"""Trainer protocol, implicit algorithm discovery, and construction factory.

Algorithm packages register themselves: importing ``rl_lib.training.ppo``
registers ``PPOTrainer`` under ``"ppo"``, and no other module needs to be
edited when a new algorithm is added. ``create_trainer`` imports the algorithm
packages on first use, so this module never names them.
"""

from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from typing import Any, TypeVar

from rl_lib.training.base import Trainer

T_Trainer = TypeVar("T_Trainer", bound=Trainer)
_TRAINER_REGISTRY: dict[str, Callable[..., Trainer]] = {}
# Packages inside this one that hold algorithms rather than shared machinery.
_NON_ALGORITHM_PACKAGES = frozenset({"callbacks"})
_algorithms_discovered = False


def register_trainer(algorithm: str, trainer_type: Callable[..., T_Trainer]) -> None:
    """Register a trainer constructor for an algorithm name.

    Algorithm packages call this from their ``__init__``, so importing the
    package is what makes the trainer discoverable.

    Args:
        algorithm: Case-insensitive algorithm key used by run configuration.
        trainer_type: Constructor returning an implementation of ``Trainer``.

    Raises:
        ValueError: If the algorithm key is empty or already registered.
    """
    key = algorithm.strip().lower()
    if not key:
        raise ValueError("algorithm name cannot be empty")
    if key in _TRAINER_REGISTRY:
        raise ValueError(f"trainer for algorithm {key!r} is already registered")
    _TRAINER_REGISTRY[key] = trainer_type


def _discover_algorithms() -> None:
    """Import every algorithm package under ``rl_lib.training`` once.

    Each package registers its own trainer on import, so this only has to
    import them; nothing here knows the algorithm names.
    """
    global _algorithms_discovered
    if _algorithms_discovered:
        return
    training_package = import_module(__package__)
    for module in iter_modules(training_package.__path__, f"{__package__}."):
        if not module.ispkg or module.name in _NON_ALGORITHM_PACKAGES:
            continue
        import_module(module.name)
    _algorithms_discovered = True


def create_trainer(algorithm: str, /, **kwargs: Any) -> Trainer:
    """Create a trainer from the registered implementation for an algorithm.

    Args:
        algorithm: Case-insensitive algorithm key, such as ``"ppo"``.
        **kwargs: Constructor arguments forwarded to the registered trainer.

    Returns:
        An instance implementing the shared ``Trainer`` interface.

    Raises:
        ValueError: If no algorithm package registered a trainer for
            ``algorithm``.
    """
    _discover_algorithms()
    key = algorithm.strip().lower()
    try:
        trainer_type = _TRAINER_REGISTRY[key]
    except KeyError as error:
        available = ", ".join(sorted(_TRAINER_REGISTRY)) or "none"
        raise ValueError(
            f"unknown training algorithm {algorithm!r}; registered algorithms: {available}"
        ) from error
    return trainer_type(**kwargs)


__all__ = ["Trainer", "create_trainer", "register_trainer"]
