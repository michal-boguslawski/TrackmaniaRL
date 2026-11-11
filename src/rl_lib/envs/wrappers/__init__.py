"""Environment wrappers package.

Exports WRAPPERS registry for make_env to instantiate Gymnasium wrappers
from config.
"""

from typing import Any

from rl_lib.envs.settings.wrappers import (
    FrameStackWrapperSettings,
    GrayscaleWrapperSettings,
    MaxAndSkipWrapperSettings,
    OffTrackPenaltyWrapperSettings,
    RecordEpisodeStatisticsWrapperSettings,
    RecordVideoWrapperSettings,
    RewardOnDoneWrapperSettings,
    WrapperSettings,
)


def __getattr__(name: str) -> Any:
    """Load the registry lazily to keep wrapper schemas dependency-light."""
    if name == "WRAPPERS":
        from rl_lib.envs.wrappers.registry import WRAPPERS_REGISTRY

        return WRAPPERS_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WRAPPERS_REGISTRY",
    "FrameStackWrapperSettings",
    "GrayscaleWrapperSettings",
    "MaxAndSkipWrapperSettings",
    "OffTrackPenaltyWrapperSettings",
    "RecordEpisodeStatisticsWrapperSettings",
    "RecordVideoWrapperSettings",
    "RewardOnDoneWrapperSettings",
    "WrapperSettings",
]
