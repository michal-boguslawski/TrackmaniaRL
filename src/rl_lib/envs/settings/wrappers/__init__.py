from typing import Annotated

from pydantic import Field

from rl_lib.envs.settings.wrappers.action_couple import CoupledGasBrakeWrapperSettings
from rl_lib.envs.settings.wrappers.framestack import FrameStackWrapperSettings
from rl_lib.envs.settings.wrappers.grayscale import GrayscaleWrapperSettings
from rl_lib.envs.settings.wrappers.max_and_skip import MaxAndSkipWrapperSettings
from rl_lib.envs.settings.wrappers.off_track_penalty import (
    OffTrackPenaltyWrapperSettings,
)
from rl_lib.envs.settings.wrappers.record_episode import (
    RecordEpisodeStatisticsWrapperSettings,
)
from rl_lib.envs.settings.wrappers.record_video import RecordVideoWrapperSettings
from rl_lib.envs.settings.wrappers.reward_on_done import RewardOnDoneWrapperSettings

WrapperSettings = Annotated[
    GrayscaleWrapperSettings
    | CoupledGasBrakeWrapperSettings
    | FrameStackWrapperSettings
    | RecordEpisodeStatisticsWrapperSettings
    | RecordVideoWrapperSettings
    | MaxAndSkipWrapperSettings
    | RewardOnDoneWrapperSettings
    | OffTrackPenaltyWrapperSettings,
    Field(discriminator="name"),
]

__all__ = [
    "CoupledGasBrakeWrapperSettings",
    "FrameStackWrapperSettings",
    "GrayscaleWrapperSettings",
    "MaxAndSkipWrapperSettings",
    "OffTrackPenaltyWrapperSettings",
    "RecordEpisodeStatisticsWrapperSettings",
    "RecordVideoWrapperSettings",
    "RewardOnDoneWrapperSettings",
    "WrapperSettings",
]
