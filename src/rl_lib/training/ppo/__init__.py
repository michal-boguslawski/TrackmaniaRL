"""PPO training components and public trainer API.

Importing this package registers ``PPOTrainer`` as the implementation for the
``"ppo"`` algorithm, which is how the trainer factory finds it.
"""

from rl_lib.training.factory import register_trainer
from rl_lib.training.ppo.trainer import PPOTrainer


register_trainer("ppo", PPOTrainer)

__all__ = ["PPOTrainer"]
