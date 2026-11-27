from rl_lib.tracking.verbosity import Verbosity, as_verbosity
from rl_lib.training.metrics.contexts import (
    OptimizationContext,
    RolloutContext,
    UpdateContext,
)
from rl_lib.training.metrics.optimization.clip_norm import calculate_clip_norms
from rl_lib.training.metrics.optimization.optimizer import calculate_learning_rates
from rl_lib.training.metrics.registry import MetricRegistry
from rl_lib.training.metrics.rollout.statistics import (
    calculate_action_statistics,
    calculate_returns_and_advantages_and_values,
)
from rl_lib.training.metrics.rollout.variance import calculate_explained_variance
from rl_lib.training.metrics.update.policy import (
    calculate_beta_concentrations,
    calculate_per_step_entropy,
)


def build_optimization_factory(
    verbosity: int | Verbosity = Verbosity.ALL,
) -> MetricRegistry[OptimizationContext]:
    """Create optimization metric producers enabled at ``verbosity``.

    Args:
        verbosity: Metric detail level; optimization diagnostics are registered
            only when diagnostics are enabled.

    Returns:
        An optimization metric registry.
    """
    optimization_factory = MetricRegistry("optimization")
    if as_verbosity(verbosity).logs_diagnostics:
        optimization_factory.register(calculate_learning_rates)
        optimization_factory.register(calculate_clip_norms)
    return optimization_factory


def build_rollout_factory(
    verbosity: int | Verbosity = Verbosity.ALL,
) -> MetricRegistry[RolloutContext]:
    """Create rollout metric producers enabled at ``verbosity``.

    Args:
        verbosity: Metric detail level; rollout reductions are registered only
            when diagnostics are enabled.

    Returns:
        A rollout metric registry.
    """
    rollout_factory = MetricRegistry("rollout")
    if as_verbosity(verbosity).logs_diagnostics:
        rollout_factory.register(calculate_action_statistics)
        rollout_factory.register(calculate_explained_variance)
        rollout_factory.register(calculate_returns_and_advantages_and_values)
    return rollout_factory


def build_update_factory(
    verbosity: int | Verbosity = Verbosity.ALL,
) -> MetricRegistry[UpdateContext]:
    """Create generic update diagnostics enabled at ``verbosity``.

    Args:
        verbosity: Metric detail level controlling whether diagnostic
            reductions are registered.

    Returns:
        An update metric registry.
    """
    update_factory = MetricRegistry("update")
    if as_verbosity(verbosity).logs_diagnostics:
        update_factory.register(calculate_beta_concentrations)
        update_factory.register(calculate_per_step_entropy)
    return update_factory
