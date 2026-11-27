import torch as T

from rl_lib.training.metrics.contexts import OptimizationContext


def calculate_learning_rates(
    optimization_context: OptimizationContext,
) -> dict[str, T.Tensor]:
    optimizer = optimization_context.optimizer
    groups = optimizer.param_groups
    reference = next(
        (parameter for group in groups for parameter in group["params"]),
        None,
    )
    if reference is None:
        return {}
    metrics = {
        f"learning_rate/{group['name']}": T.as_tensor(
            group["lr"], device=reference.device, dtype=reference.dtype
        )
        for group in groups
    }
    return metrics
