"""rl_lib: Reinforcement learning library for CarRacing and TrackMania.

This package provides a PPO implementation with:
- Vectorized environment support via Gymnasium
- CNN + temporal encoder architecture for image observations
- Beta-distribution actor for bounded continuous actions
- GAE with proper termination/truncation handling
- MLflow integration for experiment tracking
"""

# Re-export main public classes for convenient imports
from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig, load_config
from rl_lib.training.ppo import PPOTrainer
from rl_lib.training.rollout_collector import RolloutCollector

__all__ = [
    "Agent",
    "RolloutBuffer",
    "RolloutStep",
    "Network",
    "RunConfig",
    "load_config",
    "PPOTrainer",
    "RolloutCollector",
]
