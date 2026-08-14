from gymnasium.wrappers import (
    GrayscaleObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    MaxAndSkipObservation,
)


WRAPPERS = {
    "grayscale": lambda env, **kwargs: GrayscaleObservation(env, keep_dim=True),
    "frame_stack": lambda env, stack_size, **kwargs: FrameStackObservation(env, stack_size=stack_size),
    "record_episode_stats": lambda env, **kwargs: RecordEpisodeStatistics(env),
    "record_video": lambda env, video_folder, **kwargs: RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True),
    "max_and_skip": lambda env, skip, **kwargs: MaxAndSkipObservation(env, skip=skip),
}
