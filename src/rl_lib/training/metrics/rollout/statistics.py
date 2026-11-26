from rl_lib.training.metrics.contexts import RolloutContext
from rl_lib.training.utils import tensor_metrics_to_scalars


def calculate_action_statistics(context: RolloutContext) -> dict[str, float]:
    means = context.action.mean(dim=(0, 1))
    stds = context.action.std(dim=(0, 1))
    metrics = {
        **{f"action_mean_{i}": value for i, value in enumerate(means)},
        **{f"action_std_{i}": value for i, value in enumerate(stds)},
    }
    return tensor_metrics_to_scalars(metrics)


def calculate_returns_and_advantages_and_values(
    context: RolloutContext,
) -> dict[str, float]:
    metrics = {
        "mean_returns": context.returns.mean(),
        "mean_advantages": context.advantages.mean(),
        "mean_values": context.values.mean(),
    }
    return tensor_metrics_to_scalars(metrics)
