"""Environment wrapper registry and factory functions.

Provides WRAPPERS dict mapping wrapper names to factory functions that
create Gymnasium wrappers with config from WrapperSettings.
"""

from gymnasium.wrappers import (
    GrayscaleObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    MaxAndSkipObservation,
)

from rl_lib.envs.wrappers.reward_wrappers import RewardOnEpisodeEndWrapper
from rl_lib.run_config import WrapperSettings

_UNSET = object()


def _grayscale(env, keep_dim: bool | None = None, **kwargs):
    """GrayscaleObservation wrapper with default from WrapperSettings."""
    keep_dim = WrapperSettings(name="grayscale").keep_dim if keep_dim is None else keep_dim
    return GrayscaleObservation(env, keep_dim=keep_dim)


def _frame_stack(env, stack_size: int, frame_padding_type: str | None = None, **kwargs):
    """FrameStackObservation wrapper with default from WrapperSettings."""
    frame_padding_type = WrapperSettings(name="frame_stack", stack_size=stack_size).frame_padding_type if frame_padding_type is None else frame_padding_type
    return FrameStackObservation(env, stack_size=stack_size, padding_type=frame_padding_type)


def _record_episode_stats(env, stats_buffer_length: int | None = None, stats_key: str | None = None, **kwargs):
    """RecordEpisodeStatistics wrapper with defaults from WrapperSettings."""
    defaults = WrapperSettings(name="record_episode_stats")
    stats_buffer_length = defaults.stats_buffer_length if stats_buffer_length is None else stats_buffer_length
    stats_key = defaults.stats_key if stats_key is None else stats_key
    return RecordEpisodeStatistics(env, buffer_length=stats_buffer_length, stats_key=stats_key)


def _record_video(
    env,
    video_folder: str,
    episode_trigger: str | None = None,
    episode_interval: int | None = None,
    video_length: int | None = None,
    video_name_prefix: str | None = None,
    video_fps: int | None = None,
    video_disable_logger: bool | None = None,
    **kwargs,
):
    """RecordVideo wrapper with defaults from WrapperSettings."""
    defaults = WrapperSettings(name="record_video", video_folder=video_folder)
    episode_trigger = defaults.episode_trigger if episode_trigger is None else episode_trigger
    episode_interval = defaults.episode_interval if episode_interval is None else episode_interval
    video_length = defaults.video_length if video_length is None else video_length
    video_name_prefix = defaults.video_name_prefix if video_name_prefix is None else video_name_prefix
    video_fps = defaults.video_fps if video_fps is None else video_fps
    video_disable_logger = defaults.video_disable_logger if video_disable_logger is None else video_disable_logger
    return RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode: episode_trigger == "all" or episode % episode_interval == 0,
        video_length=video_length,
        name_prefix=video_name_prefix,
        fps=video_fps,
        disable_logger=video_disable_logger,
    )


def _max_and_skip(env, skip: int, **kwargs):
    """MaxAndSkipObservation wrapper."""
    return MaxAndSkipObservation(env, skip=skip)


def _reward_on_done(
    env,
    on_terminated=_UNSET,
    on_truncated=_UNSET,
    on_win=_UNSET,
    **kwargs,
):
    """RewardOnEpisodeEndWrapper with defaults from WrapperSettings."""
    defaults = WrapperSettings(name="reward_on_done")
    on_terminated = defaults.on_terminated if on_terminated is _UNSET else on_terminated
    on_truncated = defaults.on_truncated if on_truncated is _UNSET else on_truncated
    on_win = defaults.on_win if on_win is _UNSET else on_win
    return RewardOnEpisodeEndWrapper(
        env,
        on_terminated=on_terminated,
        on_truncated=on_truncated,
        on_win=on_win,
    )


WRAPPERS = {
    "grayscale": _grayscale,
    "frame_stack": _frame_stack,
    "record_episode_stats": _record_episode_stats,
    "record_video": _record_video,
    "max_and_skip": _max_and_skip,
    "reward_on_done": _reward_on_done,
}