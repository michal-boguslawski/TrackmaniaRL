from torch.distributions import Beta

from rl_lib.training.metrics.contexts import UpdateContext
from rl_lib.training.utils import tensor_metrics_to_scalars


def calculate_beta_concentrations(update_context: UpdateContext) -> dict[str, float]:
    dist = update_context.dist.base_dist
    assert isinstance(dist, Beta)
    alphas = dist.concentration1.mean(0)
    betas = dist.concentration0.mean(0)
    metrics = {
        **{f"dist/alpha_{i}": value for i, value in enumerate(alphas)},
        **{f"dist/beta_{i}": value for i, value in enumerate(betas)},
    }
    return tensor_metrics_to_scalars(metrics)


def calculate_per_step_entropy(update_context: UpdateContext) -> dict[str, float]:
    dist = update_context.dist
    entropy = dist.entropy().mean(0)
    metrics = {f"dist/entropy_{i}": value for i, value in enumerate(entropy)}
    return tensor_metrics_to_scalars(metrics)
