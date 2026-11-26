from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.utils import tensor_metrics_to_scalars


def calculate_approx_kl(update_context: PPOUpdateContext) -> dict[str, float]:
    """Compute the approximate KL divergence between two distributions.

    Args:
        log_ratios: Log probability ratios between the new and old policies.

    Returns:
        Approximate KL divergence as a scalar tensor.
    """
    assert isinstance(update_context, PPOUpdateContext)
    log_ratios = update_context.new_log_probs - update_context.old_log_probs
    approx_kls = log_ratios.exp() - 1 - log_ratios
    metrics = {"approx_kl": approx_kls.sum(-1).mean()}
    metrics.update(
        {
            f"approx_kl_{i}": mean_value
            for i, mean_value in enumerate(approx_kls.mean(0))
        }
    )
    return tensor_metrics_to_scalars(metrics)
