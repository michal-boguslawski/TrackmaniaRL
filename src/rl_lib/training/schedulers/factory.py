from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rl_lib.training.schedulers.registry import SCHEDULER_REGISTRY
from rl_lib.training.settings.scheduler import SchedulerSettings


def build_scheduler(
    optimizer: Optimizer,
    settings: SchedulerSettings | None,
    total_steps: int,
    steps_per_training: int,
) -> LRScheduler | None:
    if settings is None:
        return None

    scheduler = SCHEDULER_REGISTRY[settings.name]
    return scheduler(
        optimizer=optimizer,
        settings=settings,
        total_steps=total_steps,
        steps_per_training=steps_per_training,
    )
