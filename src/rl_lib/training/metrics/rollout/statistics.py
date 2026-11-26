from rl_lib.training.metrics.contexts import RolloutContext


def calculate_action_statistics(context: RolloutContext) -> dict[str, float]:
    means = context.action.mean(dim=(0, 1))
    stds = context.action.mean(dim=(0, 1))
    metrics = {
        **{f"action_mean_{i}": value.item() for i, value in enumerate(means)},
        **{f"action_std_{i}": value.item() for i, value in enumerate(stds)},
    }
    return metrics
