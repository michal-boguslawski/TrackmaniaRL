from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from rl_lib.networks.config import NetworkConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(StrictModel):
    algorithm: str = Field(
        default="ppo",
        min_length=1,
        description="Registered training algorithm used by the trainer factory.",
    )
    name: str = "PPO"
    total_steps: PositiveInt = 3_000_000  # vector-environment steps
    seed: int = Field(default=1, ge=0, le=2**32 - 1)
    device: str = "auto"
    clear_cuda_cache: bool = True
    log_config: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"


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


class WrapperSettings(StrictModel):
    name: Literal[
        "grayscale", "frame_stack", "record_episode_stats", "record_video",
        "max_and_skip", "reward_on_done",
    ]
    skip: PositiveInt | None = None
    stack_size: PositiveInt | None = None
    video_folder: str | None = None
    episode_trigger: Literal["all", "interval"] = "all"
    episode_interval: PositiveInt = 1
    video_length: NonNegativeInt = 0
    video_name_prefix: str = "rl-video"
    video_fps: PositiveInt | None = None
    video_disable_logger: bool = True
    stats_buffer_length: PositiveInt = 100
    stats_key: str = Field(default="episode", min_length=1)
    keep_dim: bool = True
    frame_padding_type: Literal["reset", "zero"] = "reset"
    on_terminated: FiniteFloat | None = None
    on_truncated: FiniteFloat | None = -10.0
    on_win: FiniteFloat | None = 100.0

    @model_validator(mode="after")
    def validate_wrapper_options(self):
        allowed = {
            "grayscale": {"keep_dim"},
            "frame_stack": {"stack_size", "frame_padding_type"},
            "record_episode_stats": {"stats_buffer_length", "stats_key"},
            "record_video": {
                "video_folder", "episode_trigger", "episode_interval", "video_length",
                "video_name_prefix", "video_fps", "video_disable_logger",
            },
            "max_and_skip": {"skip"},
            "reward_on_done": {"on_terminated", "on_truncated", "on_win"},
        }[self.name]
        supplied = self.model_fields_set - {"name"}
        invalid = supplied - allowed
        if invalid:
            raise ValueError(f"options {sorted(invalid)} are not supported by wrapper {self.name!r}")
        if self.name == "frame_stack" and self.stack_size is None:
            raise ValueError("frame_stack requires an explicit stack_size")
        return self


class EnvironmentSettings(StrictModel):
    id: str = "CarRacing-v3"
    num_envs: PositiveInt = 16
    skip: PositiveInt = 2
    continuous: bool = True
    normalize_rewards: bool = True
    reward_normalization_gamma: float = Field(default=0.99, ge=0, le=1)
    reward_normalization_epsilon: float = Field(default=1e-8, gt=0)
    vectorization_mode: Literal["async", "sync"] = "async"
    autoreset_mode: Literal["same_step", "next_step", "disabled"] = "same_step"
    record_video: bool = False
    video_folder: str = "./logs/videos"
    wrappers: list[WrapperSettings] = Field(default_factory=lambda: [
        WrapperSettings(name="record_episode_stats"),
        WrapperSettings(name="grayscale"),
        WrapperSettings(name="reward_on_done", on_terminated=None, on_truncated=-10.0, on_win=100.0),
        WrapperSettings(name="max_and_skip"),
    ])


class AgentSettings(StrictModel):
    stack_size: PositiveInt = 4
    observation_divisor: float = Field(default=127.5, gt=0)
    observation_offset: FiniteFloat = -1.0
    action_temperature: float = Field(default=1.0, ge=0)
    deterministic_temperature: float = Field(default=0.0, ge=0)


class RolloutSettings(StrictModel):
    buffer_size: PositiveInt = 1024
    epochs: PositiveInt = 3
    minibatch_size: PositiveInt = 512
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)


class TrainerSettings(StrictModel):
    ppo_epsilon: float = Field(default=0.2, gt=0, lt=1)
    value_clip_epsilon: float | None = Field(default=None, gt=0, lt=1)
    critic_beta: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.001, ge=0)
    entropy_decay: float = Field(default=0.995, gt=0, le=1)
    entropy_coef_min: float = Field(default=1e-4, ge=0)
    mean_reg_coef: float = Field(default=0.01, ge=0)
    advantage_normalization_strategy: Literal["batch", "global"] | None = None
    target_kl: float = Field(default=0.03, gt=0)
    kl_warmup_steps: int = Field(default=20_000, ge=0)
    kl_stop_multiplier: float = Field(default=3.0, gt=0)
    hard_stop_kl: bool = True
    backbone_lr: float = Field(default=1e-4, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    optimizer_eps: float = Field(default=1e-5, gt=0)
    optimizer_beta1: float = Field(default=0.9, ge=0, lt=1)
    optimizer_beta2: float = Field(default=0.999, ge=0, lt=1)
    optimizer_amsgrad: bool = False
    no_decay_weight_decay: float = Field(default=0.0, ge=0)
    scheduler: Literal["cosine", "linear", "none"] = "cosine"
    scheduler_min_lr: float = Field(default=1e-8, ge=0)
    scheduler_start_factor: float = Field(default=1.0, gt=0, le=1)
    scheduler_extra_iters: int = Field(default=1, ge=0)
    backbone_max_grad_norm: float = Field(default=0.5, gt=0)
    actor_max_grad_norm: float = Field(default=0.5, gt=0)
    critic_max_grad_norm: float = Field(default=0.5, gt=0)
    advantage_epsilon: float = Field(default=1e-4, gt=0)
    explained_variance_epsilon: float = Field(default=1e-8, gt=0)
    log_ratio_min: FiniteFloat = -5.0
    log_ratio_max: FiniteFloat = 5.0
    skip_nonfinite_updates: bool = True
    tanh_saturation_threshold: float = Field(default=3.5, gt=0)
    minibatch_indexing_mode: Literal["iid", "sequential"] = "iid"

    @model_validator(mode="after")
    def validate_loss_ranges(self):
        if self.log_ratio_min >= self.log_ratio_max:
            raise ValueError("trainer.log_ratio_min must be lower than log_ratio_max")
        return self


class TrackingSettings(StrictModel):
    """Which scalars reach the tracking backends.

    Attributes:
        console: Enable the console metrics logger.
        mlflow: Enable the MLflow tracking backend.
        experiment_name: MLflow experiment (defaults to the environment id).
        run_name: MLflow run name (defaults to "<run name>/<session id>").
        log_system_metrics: Let MLflow sample host metrics; only honoured at
            ``verbosity`` 2, since system metrics are diagnostics.
        verbosity: Metric detail level: 0 logs no metrics, 1 logs only the
            core training losses, 2 logs every metric.
    """

    console: bool = True
    mlflow: bool = True
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
    recording: WrapperSettings = Field(default_factory=lambda: WrapperSettings(name="record_video"))
    wrappers: list[WrapperSettings] = Field(default_factory=lambda: [
        WrapperSettings(name="record_episode_stats"),
        WrapperSettings(name="grayscale"),
        WrapperSettings(name="max_and_skip"),
    ])

    @model_validator(mode="after")
    def validate_video_wrapper(self):
        if self.recording.name != "record_video":
            raise ValueError("record_video callback recording must have name='record_video'")
        stats_wrappers = [wrapper for wrapper in self.wrappers if wrapper.name == "record_episode_stats"]
        if len(stats_wrappers) != 1:
            raise ValueError("record_video callback requires exactly one record_episode_stats wrapper")
        return self


CallbackSettings = Annotated[
    Union[
        CheckpointsCallbackConfig,
        MetricsCallbackConfig,
        EpisodeStatisticsCallbackConfig,
        EvaluationCallbackConfig,
        RecordVideoCallbackConfig,
    ],
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
            wrapper for wrapper in self.environment.wrappers
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
    agent: AgentSettings = Field(default_factory=AgentSettings)
    rollout: RolloutSettings = Field(default_factory=RolloutSettings)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    trainer: TrainerSettings = Field(default_factory=TrainerSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    callbacks: list[CallbackSettings] = Field(default_factory=list)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_callback_settings(cls, values):
        """Accept the former callback mapping while writing the list format."""
        if not isinstance(values, dict) or not isinstance(values.get("callbacks"), dict):
            return values

        old = values["callbacks"]
        migrated = []
        legacy_callbacks = (
            (
                "checkpoints",
                {"interval": old.get("checkpoint_interval", 200)},
            ),
            ("metrics", {"granularity": old.get("metrics_granularity", "batch")}),
            (
                "episode_statistics",
                {"mode": old.get("episode_statistics_mode", "mean"), "stats_key": old.get("episode_statistics_key", "episode")},
            ),
            (
                "evaluation",
                {
                    "interval": old.get("evaluation_interval", 50_000),
                    "episodes": old.get("evaluation_episodes", 10),
                    "final_episodes": old.get("final_evaluation_episodes", 1_000),
                },
            ),
            (
                "record_video",
                {
                    "interval": old.get("video_interval", 50_000),
                    "folder": old.get("video_folder", "./logs/videos"),
                    "skip": old.get("video_skip"),
                    "normalize_rewards": old.get("video_normalize_rewards", False),
                    "recording": old.get("video_recording", {"name": "record_video"}),
                    "wrappers": old.get("video_wrappers"),
                },
            ),
        )
        for name, options in legacy_callbacks:
            if old.get(name, True):
                migrated.append({"name": name, **{k: v for k, v in options.items() if v is not None}})
        values["callbacks"] = migrated
        return values

    def callback(self, name: str):
        return next((callback for callback in self.callbacks if callback.name == name), None)

    def with_runtime(self, **updates) -> RunConfig:
        """Copy this config with resolved runtime facts folded in. Validation is
        intentionally skipped: every update is either a scalar this module
        already constrains or a re-validated settings model built here."""
        return self.model_copy(update={"runtime": self.runtime.model_copy(update=updates)})

    @property
    def video_folder(self) -> str:
        callback = self.callback("record_video")
        folder = callback.folder if isinstance(callback, RecordVideoCallbackConfig) else "./logs/videos"
        return str(Path(folder) / self.runtime.session_id)

    @property
    def experiment_name(self) -> str:
        return self.tracking.experiment_name or self.environment.id

    @property
    def run_name(self) -> str:
        return self.tracking.run_name or f"{self.run.name}/{self.runtime.session_id}"

    def video_environment(self) -> EnvironmentSettings:
        """The single-environment, frame-recording environment used for
        evaluation rollouts."""
        callback = self.callback("record_video")
        if not isinstance(callback, RecordVideoCallbackConfig):
            raise ValueError("record_video callback is not configured")
        return self.environment.model_copy(update={
            "num_envs": 1,
            "skip": callback.skip or self.environment.skip,
            "vectorization_mode": "sync",
            "record_video": True,
            "video_folder": self.video_folder,
            "normalize_rewards": callback.normalize_rewards,
            "wrappers": [callback.recording, *callback.wrappers],
        })

    def evaluation_environment(self, num_envs: int | None = None) -> EnvironmentSettings:
        """Build a raw-reward evaluation environment matching training."""
        wrappers = [wrapper for wrapper in self.environment.wrappers if wrapper.name != "reward_on_done"]
        stats_wrappers = [wrapper for wrapper in wrappers if wrapper.name == "record_episode_stats"]
        if stats_wrappers:
            wrappers = [stats_wrappers[0], *[wrapper for wrapper in wrappers if wrapper is not stats_wrappers[0]]]
        else:
            wrappers.insert(0, WrapperSettings(name="record_episode_stats"))
        return self.environment.model_copy(update={
            "num_envs": self.environment.num_envs if num_envs is None else num_envs,
            "normalize_rewards": False,
            "record_video": False,
            "wrappers": wrappers,
        })

    def metrics_settings(self) -> MetricsCallbackSettings:
        callback = self.callback("metrics")
        if not isinstance(callback, MetricsCallbackConfig):
            raise ValueError("metrics callback is not configured")
        return MetricsCallbackSettings(granularity=callback.granularity)

    def episode_statistics_settings(self) -> EpisodeStatisticsCallbackSettings:
        callback = self.callback("episode_statistics")
        if not isinstance(callback, EpisodeStatisticsCallbackConfig):
            raise ValueError("episode_statistics callback is not configured")
        return EpisodeStatisticsCallbackSettings(
            mode=callback.mode,
            stats_key=callback.stats_key,
        )

    def video_settings(self) -> VideoCallbackSettings:
        callback = self.callback("record_video")
        if not isinstance(callback, RecordVideoCallbackConfig):
            raise ValueError("record_video callback is not configured")
        return VideoCallbackSettings(
            environment=self.video_environment(),
            interval=callback.interval,
            seed=self.run.seed,
        )

    def evaluation_settings(self) -> EvaluationCallbackSettings:
        callback = self.callback("evaluation")
        if not isinstance(callback, EvaluationCallbackConfig):
            raise ValueError("evaluation callback is not configured")
        return EvaluationCallbackSettings(
            environment=self.evaluation_environment(),
            interval=callback.interval,
            episodes=callback.episodes,
            final_episodes=callback.final_episodes,
            seed=self.run.seed,
        )

    @model_validator(mode="after")
    def validate_runtime_is_not_authored(self):
        if "runtime" in self.model_fields_set:
            raise ValueError("run.runtime is resolved by the entrypoint and cannot be set in YAML")
        return self

    @model_validator(mode="after")
    def validate_relationships(self):
        names = [callback.name for callback in self.callbacks]
        if len(names) != len(set(names)):
            raise ValueError("each callback may only be configured once")
        if not self.environment.continuous:
            raise ValueError("the configured PPO actor supports continuous action spaces only")
        episode_wrappers = [
            wrapper for wrapper in self.environment.wrappers
            if wrapper.name == "record_episode_stats"
        ]
        statistics_callback = self.callback("episode_statistics")
        if statistics_callback is not None and not episode_wrappers:
            raise ValueError("callbacks.episode_statistics requires a record_episode_stats environment wrapper")
        if len(episode_wrappers) > 1 and statistics_callback is not None:
            raise ValueError("callbacks.episode_statistics supports one record_episode_stats wrapper")
        if (
            statistics_callback is not None
            and episode_wrappers
            and statistics_callback.stats_key != episode_wrappers[0].stats_key
        ):
            raise ValueError("callbacks.episode_statistics_key must match environment wrapper stats_key")
        video_callback = self.callback("record_video")
        video_stats_wrappers = [
            wrapper for wrapper in video_callback.wrappers
            if wrapper.name == "record_episode_stats"
        ] if isinstance(video_callback, RecordVideoCallbackConfig) else []
        if isinstance(video_callback, RecordVideoCallbackConfig) and not video_stats_wrappers:
            raise ValueError("record_video callback requires record_episode_stats in its wrappers")
        if isinstance(video_callback, RecordVideoCallbackConfig) and len(video_stats_wrappers) > 1:
            raise ValueError("record_video callback supports one record_episode_stats wrapper")
        if self.rollout.minibatch_size > self.rollout.buffer_size * self.environment.num_envs:
            raise ValueError("rollout.minibatch_size cannot exceed buffer_size * environment.num_envs")
        if len(self.network.actor.action_low) != len(self.network.actor.action_high):
            raise ValueError("network.actor.action_low and action_high must have equal lengths")
        if any(low >= high for low, high in zip(self.network.actor.action_low, self.network.actor.action_high)):
            raise ValueError("every network.actor.action_low value must be below action_high")
        if self.trainer.scheduler_min_lr > min(self.trainer.backbone_lr, self.trainer.head_lr):
            raise ValueError("trainer.scheduler_min_lr cannot exceed either initial learning rate")
        if (
            self.trainer.scheduler == "linear"
            and self.trainer.scheduler_min_lr / max(self.trainer.backbone_lr, self.trainer.head_lr)
            > self.trainer.scheduler_start_factor
        ):
            raise ValueError("linear scheduler minimum LR is above the configured starting LR")
        return self


def load_config(path: str) -> RunConfig:
    """Load and validate a training configuration from YAML."""
    config_path = Path(path)
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
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")
    return RunConfig.model_validate(raw)
