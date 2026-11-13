"""Vectorized environment factory with configurable wrappers.

Builds Gymnasium vector environments from EnvironmentSettings, applying
wrappers in order and optionally adding reward normalization.
"""

import torch as T
from gymnasium import make_vec
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import NormalizeReward

from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.envs.wrappers.factory import make_wrapper
from rl_lib.envs.wrappers.torch_conversion import NumpyToTorch


def make_env(config: EnvironmentSettings, device: T.device | str = "cpu") -> VectorEnv:
    """Create a vectorized environment with configured wrappers.

    Args:
        config: EnvironmentSettings with id, num_envs, wrappers, etc.

    Returns:
        VectorEnv with all wrappers applied and optional reward normalization.

    Wrapper order:
    1. record_video (if config.record_video or wrapper present)
    2. All wrappers from config.wrappers in order
    3. NormalizeReward (if config.normalize_rewards)

    Video recording forces SYNC vectorization mode.
    """

    env = make_vec(
        config.id,
        num_envs=config.num_envs,
        vectorization_mode=config.vectorization_mode,
        # SAME_STEP keeps terminal actions as transitions and avoids emitting
        # a separate reset-only step on the next call to env.step().
        vector_kwargs={"autoreset_mode": config.autoreset_mode},
        render_mode=config.render_mode,
        continuous=config.continuous,
        wrappers=[make_wrapper(wrapper) for wrapper in config.wrappers],
    )

    if config.normalize_rewards:
        env = NormalizeReward(
            env,
            gamma=config.reward_normalization_gamma,
            epsilon=config.reward_normalization_epsilon,
        )
    env = NumpyToTorch(env, device=device)
    return env
