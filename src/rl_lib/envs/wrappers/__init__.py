"""Environment wrappers package.

Exports WRAPPERS registry for make_env to instantiate Gymnasium wrappers
from config.
"""

from rl_lib.envs.wrappers.registry import WRAPPERS

__all__ = ["WRAPPERS"]