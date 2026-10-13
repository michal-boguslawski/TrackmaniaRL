"""Environment wrappers package.

Exports WRAPPERS registry for make_env to instantiate Gymnasium wrappers
from config.
"""

from typing import Any

from rl_lib.envs.wrappers.settings import (
    FrameStackWrapperSettings,
    GrayscaleWrapperSettings,
    MaxAndSkipWrapperSettings,
    OffTrackPenaltyWrapperSettings,
    RecordEpisodeStatisticsWrapperSettings,
    RecordVideoWrapperSettings,
    RewardOnDoneWrapperSettings,
    WrapperSettings,
    make_wrapper_settings,
)


def __getattr__(name: str) -> Any:
    """Load the registry lazily to keep wrapper schemas dependency-light."""
    if name == "WRAPPERS":
        from rl_lib.envs.wrappers.registry import WRAPPERS

        return WRAPPERS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FrameStackWrapperSettings",
    "GrayscaleWrapperSettings",
    "MaxAndSkipWrapperSettings",
    "OffTrackPenaltyWrapperSettings",
    "RecordEpisodeStatisticsWrapperSettings",
    "RecordVideoWrapperSettings",
    "RewardOnDoneWrapperSettings",
    "WRAPPERS",
    "WrapperSettings",
    "make_wrapper_settings",
]
