from dataclasses import dataclass

import torch as T
from torch.distributions import Distribution

from rl_lib.training.metrics.contexts import UpdateContext
from rl_lib.training.ppo.config.loss import PPOActorLossSettings


@dataclass
class PPOUpdateContext(UpdateContext):
    old_log_probs: T.Tensor
    new_log_probs: T.Tensor
    advantages: T.Tensor
    actions: T.Tensor
    old_values: T.Tensor
    actor_loss_cfg: PPOActorLossSettings
