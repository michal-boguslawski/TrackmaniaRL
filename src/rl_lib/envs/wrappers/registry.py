"""Environment wrapper registry and factory functions.

Provides WRAPPERS dict mapping wrapper names to factory functions that
create Gymnasium wrappers with config from WrapperSettings.
"""

from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)

from rl_lib.envs.wrappers.off_track_penalty_wrapper import OffTrackPenaltyWrapper
from rl_lib.envs.wrappers.reward_wrappers import RewardOnEpisodeEndWrapper

WRAPPERS_REGISTRY = {
    "grayscale": GrayscaleObservation,
    "frame_stack": FrameStackObservation,
    "record_episode_stats": RecordEpisodeStatistics,
    "record_video": RecordVideo,
    "max_and_skip": MaxAndSkipObservation,
    "reward_on_done": RewardOnEpisodeEndWrapper,
    "off_track_penalty": OffTrackPenaltyWrapper,
}
