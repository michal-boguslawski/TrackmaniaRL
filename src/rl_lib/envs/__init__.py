"""Environment creation package.

Provides make_env() for building vectorized environments with configured
wrappers from run config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

    from rl_lib.run_config import EnvironmentSettings


def make_env(config: EnvironmentSettings) -> VectorEnv:
    """Create a vector environment from validated environment settings.

    Args:
        config: Environment and wrapper settings used to construct the vector environment.

    Returns:
        A configured Gymnasium vector environment.
    """
    from rl_lib.envs.make_env import make_env as make_vector_env

    return make_vector_env(config)


__all__ = ["make_env"]
