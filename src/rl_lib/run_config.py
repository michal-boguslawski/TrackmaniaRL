from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, PositiveInt, model_validator

from rl_lib.config_base import StrictModel
from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.envs.settings.wrappers import (
    RecordVideoWrapperSettings,
    WrapperSettings,
)
from rl_lib.tracking.config.metrics_logger import TrackingSettings
from rl_lib.training.ppo.config.algorithm import PPOAlgorithmConfig


class RunSettings(StrictModel):
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
    log_config: str | None = None


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
    algorithm: PPOAlgorithmConfig = Field(default_factory=PPOAlgorithmConfig)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    # runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


def load_config(path: str) -> RunConfig:
    """Load and validate a training configuration from YAML."""
    config_path = Path("src") / "rl_lib" / path
    try:
        with config_path.open("rt", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config file {config_path} must contain a YAML mapping")
    return RunConfig.model_validate(raw)
