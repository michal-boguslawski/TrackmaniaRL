from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.metrics.build import (
    build_optimization_factory,
    build_rollout_factory,
    build_update_factory,
)
from rl_lib.training.metrics.contexts import OptimizationContext, RolloutContext
from rl_lib.training.metrics.factory import MetricsFactory
from rl_lib.training.metrics.registry import MetricRegistry
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.ppo.metrics.update.policy import (
    calculate_approx_kl,
    calculate_clip_fraction,
    calculate_entropy_coef,
)


def _build_ppo_optimization_factory(
    verbosity: Verbosity,
) -> MetricRegistry[OptimizationContext]:
    optimization_factory = build_optimization_factory(verbosity)
    return optimization_factory


def _build_ppo_rollout_factory(
    verbosity: Verbosity,
) -> MetricRegistry[RolloutContext]:
    rollout_factory = build_rollout_factory(verbosity)
    return rollout_factory


def _build_ppo_update_factory(
    verbosity: Verbosity,
) -> MetricRegistry[PPOUpdateContext]:
    update_factory = build_update_factory(verbosity)
    if verbosity.logs_diagnostics:
        update_factory.register(calculate_approx_kl)
        update_factory.register(calculate_clip_fraction)
        update_factory.register(calculate_entropy_coef)
    return update_factory


def build_ppo_metrics_factory(
    verbosity: int | Verbosity = Verbosity.ALL,
) -> MetricsFactory:
    """Build PPO metric producers enabled at the requested verbosity.

    Diagnostic registries are empty below ``Verbosity.ALL``. When hard KL
    stopping is configured, ``PPOTrainer`` computes only the scalar KL required
    by the update loop outside these registries.

    Args:
        verbosity: Metric detail level (0, 1, or 2).

    Returns:
        Metric registries for rollout, update, and optimization calculations.
    """
    level = as_verbosity(verbosity)
    return MetricsFactory(
        optimization=_build_ppo_optimization_factory(level),
        rollout=_build_ppo_rollout_factory(level),
        update=_build_ppo_update_factory(level),
    )
