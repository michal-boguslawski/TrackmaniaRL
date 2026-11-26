from dataclasses import dataclass

from rl_lib.training.metrics.contexts import (
    OptimizationContext,
    RolloutContext,
    UpdateContext,
)
from rl_lib.training.metrics.registry import MetricRegistry


@dataclass
class MetricsFactory:
    optimization: MetricRegistry[OptimizationContext]
    rollout: MetricRegistry[RolloutContext]
    update: MetricRegistry[UpdateContext]
