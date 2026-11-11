from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PositiveInt, model_validator

from rl_lib.config_base import StrictModel
from rl_lib.envs.settings.wrappers import (
    RecordVideoWrapperSettings,
    WrapperSettings,
)

# from rl_lib.training.ppo.config import PPONetworkConfig
from rl_lib.training.settings import (
    CosineSchedulerSettings,
    LinearSchedulerSettings,
    TrainerSettings,
)


class RunSettings(StrictModel):
    experiment_name: str | None = None
    total_steps: PositiveInt = 3_000_000  # vector-environment steps
    seed: int = Field(default=1, ge=0, le=2**32 - 1)
    device: str = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    channels_last: bool = Field(
        default=True,
        description=(
            "Store CNN convolution weights in channels_last (NHWC) memory "
            "format (CUDA only). cuDNN's fastest kernels are NHWC-native; this "
            "removes the per-call NCHW<->NHWC conversion kernels. The permuted "
            "NHWC observations already arrive with channels_last strides."
        ),
    )


class RuntimeSettings(StrictModel):
    """Process-level facts resolved at startup. Never authored in YAML: the
    entrypoint fills this in via RunConfig.with_runtime once logging, the device
    and the environment are known."""

    session_id: str = ""
    config_path: str | None = None
    device: str = "cpu"
    observation_shape: list[int] = Field(default_factory=list)
    action_shape: list[int] = Field(default_factory=list)
    video_folder: str = ""
    total_steps_unit: str = "vector-environment steps"
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None


class EnvironmentSettings(StrictModel):
    id: str = "CarRacing-v3"
    num_envs: PositiveInt = 8
    skip: PositiveInt = 2
    continuous: bool = True
    normalize_rewards: bool = True
    reward_normalization_gamma: float = Field(default=0.99, ge=0, le=1)
    reward_normalization_epsilon: float = Field(default=1e-8, gt=0)
    vectorization_mode: Literal["async", "sync"] = "async"
    autoreset_mode: Literal["same_step", "next_step", "disabled"] = "same_step"
    wrappers: list[WrapperSettings] = Field(default_factory=lambda: [list])


class TrackingSettings(StrictModel):
    """Which scalars reach the tracking backends.

    Attributes:
        console: Enable the console metrics logger.
        mlflow: Enable the MLflow tracking backend.
        local_artifacts: Enable local artifact logging. Artifacts are stored in logs folder.
        experiment_name: MLflow experiment (defaults to the environment id).
        run_name: MLflow run name (defaults to "<run name>/<session id>").
        log_system_metrics: Let MLflow sample host metrics; only honoured at
            ``verbosity`` 2, since system metrics are diagnostics.
        verbosity: Metric detail level: 0 logs no metrics, 1 logs only the
            core training losses, 2 logs every metric.
    """

    console: bool = True
    mlflow: bool = True
    local_artifacts: bool = False
    experiment_name: str | None = None
    run_name: str | None = None
    log_system_metrics: bool = True
    verbosity: Literal[0, 1, 2] = Field(
        default=2,
        description="Metric detail: 0 none, 1 core training losses only, 2 everything.",
    )


class CheckpointsCallbackConfig(StrictModel):
    name: Literal["checkpoints"]
    interval: PositiveInt = 200


class MetricsCallbackConfig(StrictModel):
    name: Literal["metrics"]
    granularity: Literal["minibatch", "epoch", "batch"] = "batch"


class EpisodeStatisticsCallbackConfig(StrictModel):
    name: Literal["episode_statistics"]
    mode: Literal["step", "mean"] = "mean"
    stats_key: str = Field(default="episode", min_length=1)


class EvaluationCallbackConfig(StrictModel):
    name: Literal["evaluation"]
    interval: PositiveInt = 50_000
    episodes: PositiveInt = 10
    final_episodes: PositiveInt = 1_000


class RecordVideoCallbackConfig(StrictModel):
    name: Literal["record_video"]
    interval: PositiveInt = 50_000
    folder: str = "./logs/videos"
    skip: PositiveInt | None = None
    normalize_rewards: bool = False
    recording: RecordVideoWrapperSettings = Field(
        default_factory=RecordVideoWrapperSettings
    )
    wrappers: list[WrapperSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_video_wrapper(self):
        if self.recording.name != "record_video":
            raise ValueError(
                "record_video callback recording must have name='record_video'"
            )
        stats_wrappers = [
            wrapper
            for wrapper in self.wrappers
            if wrapper.name == "record_episode_stats"
        ]
        if len(stats_wrappers) != 1:
            raise ValueError(
                "record_video callback requires exactly one record_episode_stats wrapper"
            )
        return self


CallbackSettings = Annotated[
    CheckpointsCallbackConfig
    | MetricsCallbackConfig
    | EpisodeStatisticsCallbackConfig
    | EvaluationCallbackConfig
    | RecordVideoCallbackConfig,
    Field(discriminator="name"),
]


class MetricsCallbackSettings(StrictModel):
    granularity: Literal["minibatch", "epoch", "batch"]


class EpisodeStatisticsCallbackSettings(StrictModel):
    mode: Literal["step", "mean"]
    stats_key: str = Field(default="episode", min_length=1)


class EvaluationCallbackSettings(StrictModel):
    """Resolved settings for scheduled and final policy evaluation."""

    environment: EnvironmentSettings
    interval: PositiveInt
    episodes: PositiveInt
    final_episodes: PositiveInt
    seed: int


class VideoCallbackSettings(StrictModel):
    """Everything the video callback needs, resolved from the environment,
    callback and run sections. The evaluation environment is a single
    synchronous env that records frames."""

    environment: EnvironmentSettings
    interval: PositiveInt
    seed: int

    @property
    def stats_key(self) -> str:
        stats_wrappers = [
            wrapper
            for wrapper in self.environment.wrappers
            if wrapper.name == "record_episode_stats"
        ]
        if len(stats_wrappers) != 1:
            raise ValueError(
                "the video environment needs exactly one record_episode_stats wrapper to read episode stats"
            )
        return stats_wrappers[0].stats_key


class RunConfig(StrictModel):
    run: RunSettings = Field(default_factory=RunSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    # network: NetworkSettings = Field(default_factory=PPONetworkConfig)
    trainer: TrainerSettings = Field(default_factory=TrainerSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    callbacks: list[CallbackSettings] = Field(default_factory=list)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
