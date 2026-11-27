import torch as T

from rl_lib.training.metrics.contexts import RolloutContext


def calculate_explained_variance(
    rollout_context: RolloutContext,
) -> dict[str, T.Tensor]:
    variance_returns = rollout_context.returns.var()

    explained_variance = (
        1 - (rollout_context.values - rollout_context.returns).var() / variance_returns
    )
    return {"explained_variance": explained_variance}
