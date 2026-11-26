from rl_lib.training.metrics.contexts import OptimizationContext


def calculate_learning_rates(
    optimization_context: OptimizationContext,
) -> dict[str, float]:
    optimizer = optimization_context.optimizer
    metrics = {f"learning_rate/{pg['name']}": pg["lr"] for pg in optimizer.param_groups}
    return metrics
