"""Metric conversion helpers shared by PPO training components."""

import torch as T
from torch.profiler import record_function


def _tensor_metrics_to_scalars(metrics: dict[str, T.Tensor]) -> dict[str, float]:
    """Transfer scalar tensor metrics to Python in one device synchronization.

    The final ``.cpu()`` blocks until the queued work on the accelerator has
    finished, so on CUDA this is a pipeline stall, not just a copy. It runs at
    least once per minibatch whenever diagnostics are logged.
    """
    if not metrics:
        return {}
    keys = tuple(metrics)
    with record_function("transfer/metrics_device_sync"):
        values = T.stack([metrics[key].detach().reshape(()) for key in keys])
        scalars = values.cpu().tolist()
    return dict(zip(keys, scalars, strict=True))


def calc_approx_kl_divergence(log_ratios: T.Tensor) -> T.Tensor:
    """Compute the approximate KL divergence between two distributions.

    Args:
        log_ratios: Log probability ratios between the new and old policies.

    Returns:
        Approximate KL divergence as a scalar tensor.
    """
    log_ratios_total = log_ratios.sum(-1)
    return (log_ratios_total.exp() - 1 - log_ratios_total).mean()
