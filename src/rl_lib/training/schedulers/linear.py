from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR

from rl_lib.training.settings.scheduler import LinearSchedulerSettings


def build_lr_scheduler(
    optimizer: Optimizer,
    settings: LinearSchedulerSettings,
    total_steps: int,
    steps_per_training: int,
):
    total_iters = int(total_steps / steps_per_training * settings.end_fraction)
    return LinearLR(
        optimizer,
        settings.start_factor,
        settings.end_factor,
        total_iters=total_iters,
    )
