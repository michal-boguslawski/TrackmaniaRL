from rl_lib.training.metrics.build import (
    build_optimization_factory,
    build_rollout_factory,
    build_update_factory,
)
from rl_lib.training.metrics.contexts import OptimizationContext, RolloutContext
from rl_lib.training.metrics.factory import MetricsFactory
from rl_lib.training.metrics.registry import MetricRegistry
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.ppo.metrics.update.policy import calculate_approx_kl


def _build_ppo_optimization_factory() -> MetricRegistry[OptimizationContext]:
    optimization_factory = build_optimization_factory()
    return optimization_factory


def _build_ppo_rollout_factory() -> MetricRegistry[RolloutContext]:
    rollout_factory = build_rollout_factory()
    return rollout_factory


def _build_ppo_update_factory() -> MetricRegistry[PPOUpdateContext]:
    update_factory = build_update_factory()
    update_factory.register(calculate_approx_kl)
    return update_factory


def build_ppo_metrics_factory() -> MetricsFactory:
    return MetricsFactory(
        optimization=_build_ppo_optimization_factory(),
        rollout=_build_ppo_rollout_factory(),
        update=_build_ppo_update_factory(),
    )
