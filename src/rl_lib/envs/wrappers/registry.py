from gymnasium.wrappers import (
    GrayscaleObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    MaxAndSkipObservation,
)

from rl_lib.envs.wrappers.reward_wrappers import RewardOnEpisodeEndWrapper


WRAPPERS = {
    "grayscale": lambda env, **kwargs: GrayscaleObservation(env, keep_dim=True),
    "frame_stack": lambda env, stack_size, **kwargs: FrameStackObservation(env, stack_size=stack_size),
    "record_episode_stats": lambda env, **kwargs: RecordEpisodeStatistics(env),
    "record_video": lambda env, video_folder, **kwargs: RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True),
    "max_and_skip": lambda env, skip, **kwargs: MaxAndSkipObservation(env, skip=skip),
    "reward_on_done": lambda env, on_terminated=None, on_truncated=-10, on_win=100, **kwargs: (
        RewardOnEpisodeEndWrapper(env, on_terminated=on_terminated, on_truncated=on_truncated, on_win=on_win)
    ),
}
