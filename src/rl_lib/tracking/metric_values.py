"""Types and conversion helpers for scalar metric values."""

from collections.abc import Mapping

import torch as T
from torch.profiler import record_function


def metric_values_to_scalars(
    metrics: Mapping[str, T.Tensor],
) -> dict[str, float]:
    """Convert scalar tensor metrics to Python floats with one device transfer.

    Tensor values are detached and stacked before the host copy, so a batch of
    GPU metrics causes one synchronization rather than one per metric.

    Args:
        metrics: Metric names mapped to zero-dimensional tensors.

    Returns:
        Metric names mapped to Python floats.

    Raises:
        ValueError: If a tensor metric is not scalar, requires gradients, or
            differs in device or dtype from the other metrics.
    """
    if not metrics:
        return {}

    names = tuple(metrics)
    values = tuple(metrics.values())
    reference = values[0]
    for name, value in metrics.items():
        if value.ndim != 0:
            raise ValueError(
                f"Metric {name!r} must be a zero-dimensional tensor, "
                f"got shape {tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(f"Metric {name!r} must be detached before logging")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                "Tensor metrics must share a device and dtype; "
                f"metric {name!r} has {value.device}/{value.dtype}, while "
                f"the first tensor metric has "
                f"{reference.device}/{reference.dtype}"
            )

    with record_function("transfer/metrics_device_sync"):
        host_values = T.stack(values).cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(names, host_values, strict=True)
    }
