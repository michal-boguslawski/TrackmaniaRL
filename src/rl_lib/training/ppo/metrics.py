"""Metric conversion helpers shared by PPO training components."""

import torch as T


def _tensor_metrics_to_scalars(metrics: dict[str, T.Tensor]) -> dict[str, float]:
    """Transfer scalar tensor metrics to Python in one device synchronization."""
    if not metrics:
        return {}
    keys = tuple(metrics)
    values = T.stack([metrics[key].detach().reshape(()) for key in keys])
    scalars = values.cpu().tolist()
    return dict(zip(keys, scalars, strict=True))
