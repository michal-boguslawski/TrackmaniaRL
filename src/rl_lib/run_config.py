from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel
from rl_lib.envs.settings.env import EnvironmentSettings
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


class CheckpointsCallbackConfig(StrictModel):
    name: Literal["checkpoints"]
    interval: PositiveInt = 200


class EvaluationCallbackConfig(StrictModel):
    name: Literal["evaluation"]
    interval: PositiveInt = 50_000
    episodes: PositiveInt = 10
    final_episodes: PositiveInt = 1_000


class RecordVideoCallbackConfig(StrictModel):
    name: Literal["record_video"]
    interval: PositiveInt = 50_000
    folder: str = "./logs/videos"


class EvaluationCallbackSettings(StrictModel):
    """Resolved settings for scheduled and final policy evaluation."""

    environment: EnvironmentSettings
    interval: PositiveInt
    episodes: PositiveInt
    final_episodes: PositiveInt
    seed: int


class RunConfig(StrictModel):
    run: RunSettings = Field(default_factory=RunSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    algorithm: PPOAlgorithmConfig = Field(default_factory=PPOAlgorithmConfig)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)


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
