from collections.abc import Callable

from torch.optim.lr_scheduler import LRScheduler

from rl_lib.training.schedulers.linear import build_lr_scheduler

SCHEDULER_REGISTRY: dict[str, Callable[..., LRScheduler]] = {
    "linear": build_lr_scheduler,
}
