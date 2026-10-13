from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from rl_lib.config_base import StrictModel
from rl_lib.envs.wrappers.settings import (
    RecordVideoWrapperSettings,
    WrapperSettings,
    make_wrapper_settings,
)
from rl_lib.networks.config import NetworkConfig
from rl_lib.training.settings import (
    CosineSchedulerSettings,
    LinearSchedulerSettings,
    TrainerSettings,
)


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
    pretrained_checkpoint: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional local model or state-dict checkpoint to load before training. "
            "Optimizer state and training progress are not restored."
        ),
    )
    pretrained_mlflow_run_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional MLflow run ID or runs:/ URI used to initialize network weights."
        ),
    )
    pretrained_mlflow_checkpoint_step: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Optional periodic checkpoint step from pretrained_mlflow_run_id. "
            "When omitted, the MLflow run's final_model artifact is loaded."
        ),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    cudnn_benchmark: bool = Field(
        default=True,
        description=(
            "Enable cuDNN's autotuner for convolution algorithms (CUDA only). "
            "Convolution shapes are static across PPO updates, so the cached "
            "algorithm choice amortizes over the whole run."
        ),
    )
    channels_last: bool = Field(
        default=True,
        description=(
            "Store CNN convolution weights in channels_last (NHWC) memory "
            "format (CUDA only). cuDNN's fastest kernels are NHWC-native; this "
            "removes the per-call NCHW<->NHWC conversion kernels. The permuted "
            "NHWC observations already arrive with channels_last strides."
        ),
    )
    torch_compile: bool = Field(
        default=False,
        description=(
            "Wrap the policy/value Network in torch.compile. Fuses elementwise "
            "chains and reduces kernel-launch overhead; with mode "
            "'reduce-overhead' on CUDA it also captures CUDA graphs. Adds a "
            "one-time compilation cost per input shape."
        ),
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="Mode passed to torch.compile when torch_compile is enabled.",
    )
    cnn_autocast: Literal["none", "bf16"] = Field(
        default="none",
        description=(
            "Autocast only the CNN feature extractor to BF16 on CUDA. Temporal, "
            "actor/critic heads, distribution math, and PPO losses remain FP32. "
            "Requires a CUDA GPU with BF16 support."
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
        make_wrapper_settings("record_episode_stats"),
        make_wrapper_settings("grayscale"),
        make_wrapper_settings("reward_on_done", on_terminated=None, on_truncated=-10.0, on_win=100.0),
        make_wrapper_settings("max_and_skip"),
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
    recording: RecordVideoWrapperSettings = Field(default_factory=RecordVideoWrapperSettings)
    wrappers: list[WrapperSettings] = Field(default_factory=lambda: [
        make_wrapper_settings("record_episode_stats"),
        make_wrapper_settings("grayscale"),
        make_wrapper_settings("max_and_skip"),
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
    def default_trainer_name_from_run_algorithm(cls, values):
        """Keep the trainer identity aligned with the selected run algorithm."""
        if not isinstance(values, dict):
            return values
        run = values.get("run", {})
        trainer = values.get("trainer", {})
        if not isinstance(run, dict) or not isinstance(trainer, dict):
            return values
        trainer = dict(trainer)
        trainer.setdefault("name", run.get("algorithm", "ppo"))
        values = dict(values)
        values["trainer"] = trainer
        return values

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
            wrappers.insert(0, make_wrapper_settings("record_episode_stats"))
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
        if self.trainer.name != self.run.algorithm:
            raise ValueError("trainer.name must match run.algorithm")
        if (
            self.run.pretrained_checkpoint is not None
            and self.run.pretrained_mlflow_run_id is not None
        ):
            raise ValueError(
                "configure only one of run.pretrained_checkpoint and "
                "run.pretrained_mlflow_run_id"
            )
        if (
            self.run.pretrained_mlflow_checkpoint_step is not None
            and self.run.pretrained_mlflow_run_id is None
        ):
            raise ValueError(
                "run.pretrained_mlflow_checkpoint_step requires "
                "run.pretrained_mlflow_run_id"
            )
        if (
            self.run.pretrained_mlflow_run_id is not None
            and not self.run.pretrained_mlflow_run_id.strip()
        ):
            raise ValueError("run.pretrained_mlflow_run_id cannot be blank")
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
        optimizer = self.trainer.optimizer
        initial_lrs = (
            optimizer.cnn_lr or optimizer.backbone_lr,
            optimizer.sequence_encoder_lr or optimizer.backbone_lr,
            optimizer.head_lr,
        )
        scheduler = self.trainer.scheduler
        if isinstance(scheduler, (CosineSchedulerSettings, LinearSchedulerSettings)):
            minimum_lr = scheduler.min_lr if isinstance(scheduler, LinearSchedulerSettings) else scheduler.eta_min
            if minimum_lr > min(initial_lrs):
                raise ValueError("trainer.scheduler minimum learning rate cannot exceed any initial learning rate")
        if isinstance(scheduler, LinearSchedulerSettings):
            if scheduler.min_lr / max(initial_lrs) > scheduler.start_factor:
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
