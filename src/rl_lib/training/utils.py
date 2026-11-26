import torch as T
from torch.optim import Adam, AdamW, Optimizer
from torch.profiler import record_function


def build_optimizer(params: list[dict], optimizer_name: str, eps: float) -> Optimizer:
    opt_cls = Adam if optimizer_name == "Adam" else AdamW
    return opt_cls(params, eps=eps)


def tensor_metrics_to_scalars(metrics: dict[str, T.Tensor]) -> dict[str, float]:
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
