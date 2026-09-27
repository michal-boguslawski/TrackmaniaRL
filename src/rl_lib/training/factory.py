"""Trainer protocol, implicit algorithm discovery, and construction factory.

Algorithm packages register themselves: importing ``rl_lib.training.ppo``
registers ``PPOTrainer`` under ``"ppo"``, and no other module needs to be
edited when a new algorithm is added. ``create_trainer`` imports the algorithm
packages on first use, so this module never names them.
"""

from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from typing import Any, Protocol, TypeVar

from gymnasium import Env
from numpy.typing import NDArray
import torch as T

from rl_lib.run_config import RolloutSettings, RunSettings


class Trainer(Protocol):
    """Interface required by the rollout collector and training entrypoint."""

    @property
    def stack_size(self) -> int:
        """Temporal observation window size."""
        ...

    @property
    def device(self) -> T.device:
        """Device used for policy and value computations."""
        ...

    def setup_train(self, run: RunSettings, rollout: RolloutSettings) -> None:
        """Initialize training schedules from run and rollout settings."""
        ...

    def act(
        self, observation: T.Tensor, done: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """Choose actions and return their log probabilities and values."""
        ...

    def step_env(
        self,
        env: Env,
        state: NDArray,
        done: T.Tensor,
        temperature: float | None = None,
    ) -> tuple:
        """Step the environment using the current policy."""
        ...

    def bootstrap_value(self, observation: T.Tensor) -> T.Tensor:
        """Estimate values for truncated transitions."""
        ...

    def train(
        self,
        batch: dict[str, T.Tensor],
        epochs: int,
        minibatch_size: int,
        training_step: int,
    ) -> None:
        """Update the policy/value model from a collected batch."""
        ...

    def config(self) -> dict[str, int | float | str]:
        """Return trainer settings suitable for experiment logging."""
        ...

    def network_config(self) -> dict[str, int | float | str]:
        """Return network settings suitable for experiment logging."""
        ...


T_Trainer = TypeVar("T_Trainer", bound=Trainer)
_TRAINER_REGISTRY: dict[str, Callable[..., Trainer]] = {}
# Packages inside this one that hold algorithms rather than shared machinery.
_NON_ALGORITHM_PACKAGES = frozenset({"callbacks"})
_algorithms_discovered = False


def register_trainer(
    algorithm: str, trainer_type: Callable[..., T_Trainer]
) -> None:
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
