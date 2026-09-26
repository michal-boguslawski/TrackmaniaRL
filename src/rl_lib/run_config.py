from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from rl_lib.networks.config import NetworkConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(StrictModel):
    name: str = "PPO"
    total_steps: PositiveInt = 3_000_000  # vector-environment steps
    seed: int = Field(default=1, ge=0, le=2**32 - 1)
    device: str = "auto"
    clear_cuda_cache: bool = True
    log_config: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"


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
    console: bool = True
    mlflow: bool = True
    experiment_name: str | None = None
    run_name: str | None = None
    log_system_metrics: bool = True


class CallbackSettings(StrictModel):
    checkpoints: bool = True
    checkpoint_interval: PositiveInt = 200
    checkpoint_folder: str = "./logs/checkpoints"
    metrics: bool = True
    metrics_granularity: Literal["minibatch", "epoch", "batch"] = "batch"
    episode_statistics: bool = True
    episode_statistics_mode: Literal["step", "mean"] = "mean"
    episode_statistics_key: str = Field(default="episode", min_length=1)
    record_video: bool = True
    video_interval: PositiveInt = 50_000
    video_folder: str = "./logs/videos"
    video_skip: PositiveInt | None = None
    video_normalize_rewards: bool = False
    video_recording: WrapperSettings = Field(default_factory=lambda: WrapperSettings(name="record_video"))
    video_wrappers: list[WrapperSettings] = Field(default_factory=lambda: [
        WrapperSettings(name="record_episode_stats"),
        WrapperSettings(name="grayscale"),
        WrapperSettings(name="max_and_skip"),
    ])

    @model_validator(mode="after")
    def validate_video_wrapper(self):
        if self.video_recording.name != "record_video":
            raise ValueError("callbacks.video_recording must have name='record_video'")
        return self


class RunConfig(StrictModel):
    run: RunSettings = Field(default_factory=RunSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    rollout: RolloutSettings = Field(default_factory=RolloutSettings)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    trainer: TrainerSettings = Field(default_factory=TrainerSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    callbacks: CallbackSettings = Field(default_factory=CallbackSettings)

    @model_validator(mode="after")
    def validate_relationships(self):
        if not self.environment.continuous:
            raise ValueError("the configured PPO actor supports continuous action spaces only")
        episode_wrappers = [
            wrapper for wrapper in self.environment.wrappers
            if wrapper.name == "record_episode_stats"
        ]
        if self.callbacks.episode_statistics and not episode_wrappers:
            raise ValueError("callbacks.episode_statistics requires a record_episode_stats environment wrapper")
        if len(episode_wrappers) > 1 and self.callbacks.episode_statistics:
            raise ValueError("callbacks.episode_statistics supports one record_episode_stats wrapper")
        if (
            self.callbacks.episode_statistics
            and episode_wrappers
            and self.callbacks.episode_statistics_key != episode_wrappers[0].stats_key
        ):
            raise ValueError("callbacks.episode_statistics_key must match environment wrapper stats_key")
        video_stats_wrappers = [
            wrapper for wrapper in self.callbacks.video_wrappers
            if wrapper.name == "record_episode_stats"
        ]
        if self.callbacks.record_video and not video_stats_wrappers:
            raise ValueError("callbacks.record_video requires record_episode_stats in video_wrappers")
        if self.callbacks.record_video and len(video_stats_wrappers) > 1:
            raise ValueError("callbacks.record_video supports one record_episode_stats video wrapper")
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
    from pathlib import Path

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
