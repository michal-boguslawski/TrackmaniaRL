"""Wrapper-specific configuration models and their discriminated union."""

from typing import Annotated, Literal, Union

from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt

from rl_lib.config_base import StrictModel


class GrayscaleWrapperSettings(StrictModel):
    """Settings for converting observations to grayscale.

    Attributes:
        keep_dim: Whether to retain a one-channel dimension in the output.
    """

    name: Literal["grayscale"] = Field(
        default="grayscale", description="Select the grayscale observation wrapper."
    )
    keep_dim: bool = Field(default=True, description="Keep the grayscale channel dimension.")


class FrameStackWrapperSettings(StrictModel):
    """Settings for stacking consecutive observations.

    Attributes:
        stack_size: Number of consecutive observations to combine.
        frame_padding_type: Strategy used to pad the initial frame history.
    """

    name: Literal["frame_stack"] = Field(
        default="frame_stack", description="Select the frame-stack wrapper."
    )
    stack_size: PositiveInt = Field(description="Number of consecutive frames to stack.")
    frame_padding_type: Literal["reset", "zero"] = Field(
        default="reset", description="How to pad the initial observation history."
    )


class RecordEpisodeStatisticsWrapperSettings(StrictModel):
    """Settings for tracking episode rewards and lengths.

    Attributes:
        stats_buffer_length: Number of recent episode results to retain.
        stats_key: Info key containing episode statistics.
    """

    name: Literal["record_episode_stats"] = Field(
        default="record_episode_stats",
        description="Select the episode-statistics wrapper.",
    )
    stats_buffer_length: PositiveInt = Field(
        default=100, description="Number of recent episode results to retain."
    )
    stats_key: str = Field(
        default="episode", min_length=1, description="Info key used for episode statistics."
    )


class RecordVideoWrapperSettings(StrictModel):
    """Settings for recording environment episodes to video.

    Attributes:
        video_folder: Output directory, inherited from environment settings if unset.
        episode_trigger: Whether to record all episodes or an interval.
        episode_interval: Interval used by the interval trigger.
        video_length: Maximum recorded episode length.
        video_name_prefix: Prefix used for video filenames.
        video_fps: Optional recorded video frame rate.
        video_disable_logger: Whether to disable Gymnasium's video logger.
    """

    name: Literal["record_video"] = Field(
        default="record_video", description="Select the video-recording wrapper."
    )
    video_folder: str | None = Field(
        default=None, description="Output directory; defaults to the environment video folder."
    )
    episode_trigger: Literal["all", "interval"] = Field(
        default="all", description="Record all episodes or episodes at an interval."
    )
    episode_interval: PositiveInt = Field(
        default=1, description="Episode interval when episode_trigger is interval."
    )
    video_length: NonNegativeInt = Field(
        default=0, description="Maximum recorded episode length; zero records complete episodes."
    )
    video_name_prefix: str = Field(
        default="rl-video", description="Prefix used for recorded video filenames."
    )
    video_fps: PositiveInt | None = Field(
        default=None, description="Optional playback frame rate override."
    )
    video_disable_logger: bool = Field(
        default=True, description="Disable Gymnasium's video recorder logger."
    )


class MaxAndSkipWrapperSettings(StrictModel):
    """Settings for repeating actions and max-pooling observations.

    Attributes:
        skip: Frames per action, defaulting to the environment's skip setting.
    """

    name: Literal["max_and_skip"] = Field(
        default="max_and_skip", description="Select the max-and-skip wrapper."
    )
    skip: PositiveInt | None = Field(
        default=None, description="Frames per action; defaults to EnvironmentSettings.skip."
    )


class RewardOnDoneWrapperSettings(StrictModel):
    """Reward overrides applied when an episode terminates or truncates.

    Attributes:
        on_terminated: Optional reward assigned on termination.
        on_truncated: Optional reward assigned on truncation.
        on_win: Optional reward assigned on a successful finish.
    """

    name: Literal["reward_on_done"] = Field(
        default="reward_on_done", description="Select the episode-end reward wrapper."
    )
    on_terminated: FiniteFloat | None = Field(
        default=None, description="Optional reward assigned on termination."
    )
    on_truncated: FiniteFloat | None = Field(
        default=-10.0, description="Optional reward assigned on truncation."
    )
    on_win: FiniteFloat | None = Field(
        default=100.0, description="Optional reward assigned on a successful finish."
    )


class OffTrackPenaltyWrapperSettings(StrictModel):
    """Settings for penalizing distance beyond the track edge.

    Attributes:
        penalty_weight: Maximum magnitude of the per-step penalty.
    """

    name: Literal["off_track_penalty"] = Field(
        default="off_track_penalty", description="Select the off-track penalty wrapper."
    )
    penalty_weight: FiniteFloat = Field(
        default=0.05,
        ge=0.0,
        description="Maximum magnitude of the per-step penalty beyond the road edge.",
    )


WrapperSettings = Annotated[
    Union[
        GrayscaleWrapperSettings,
        FrameStackWrapperSettings,
        RecordEpisodeStatisticsWrapperSettings,
        RecordVideoWrapperSettings,
        MaxAndSkipWrapperSettings,
        RewardOnDoneWrapperSettings,
        OffTrackPenaltyWrapperSettings,
    ],
    Field(discriminator="name"),
]


def make_wrapper_settings(name: str, **options: object) -> WrapperSettings:
    """Build the wrapper-specific settings model selected by ``name``.

    Args:
        name: Registered wrapper name.
        **options: Wrapper-specific settings passed to that wrapper model.

    Returns:
        A validated wrapper-specific settings instance.

    Raises:
        ValueError: If ``name`` is not a supported wrapper setting.
    """
    settings_type_by_name = {
        "grayscale": GrayscaleWrapperSettings,
        "frame_stack": FrameStackWrapperSettings,
        "record_episode_stats": RecordEpisodeStatisticsWrapperSettings,
        "record_video": RecordVideoWrapperSettings,
        "max_and_skip": MaxAndSkipWrapperSettings,
        "reward_on_done": RewardOnDoneWrapperSettings,
        "off_track_penalty": OffTrackPenaltyWrapperSettings,
    }
    try:
        settings_type = settings_type_by_name[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment wrapper: {name}") from exc
    return settings_type(name=name, **options)
