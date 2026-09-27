"""Environment creation package.

Provides make_env() for building vectorized environments with configured
wrappers from run config.
"""

from rl_lib.envs.make_env import make_env

__all__ = ["make_env"]