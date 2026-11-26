from rl_lib.training.metrics.contexts import (
    OptimizationContext,
    RolloutContext,
    UpdateContext,
)
from rl_lib.training.metrics.registry import MetricRegistry
from rl_lib.training.metrics.rollout.statistics import calculate_action_statistics
from rl_lib.training.metrics.rollout.variance import calculate_explained_variance
from rl_lib.training.metrics.update.policy import calculate_beta_concentrations


def build_optimization_factory() -> MetricRegistry[OptimizationContext]:
    optimization_factory = MetricRegistry("optimization")
    return optimization_factory


def build_rollout_factory() -> MetricRegistry[RolloutContext]:
    rollout_factory = MetricRegistry("rollout")
    rollout_factory.register(calculate_action_statistics)
    rollout_factory.register(calculate_explained_variance)
    return rollout_factory


def build_update_factory() -> MetricRegistry[UpdateContext]:
    update_factory = MetricRegistry("update")
    update_factory.register(calculate_beta_concentrations)
    return update_factory
