"""Configuration for optional PyTorch performance profiling.

Profiling settings are kept separate from the PPO run configuration so that
diagnostic capture can be adjusted without changing training hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt


class ProfilingConfig(BaseModel):
    """Settings controlling a short PyTorch profiler capture.

    The schedule counts vector-environment iterations, not individual
    environments or optimizer minibatches.

    Attributes:
        enabled: Whether profiling is enabled unless overridden by the CLI.
        wait_steps: Scheduled iterations to skip before warmup. ``None`` aligns
            capture with the first PPO update in the run.
        warmup_steps: Iterations used to warm up the profiler without recording.
        active_steps: Number of iterations recorded in each profiling window.
        repeat: Number of scheduled profiling windows to capture.
        output_dir: Directory where profiler trace files are written.
        record_shapes: Include tensor input shapes in operator events.
        profile_memory: Include tensor memory allocation events.
        with_stack: Include Python stack traces in operator events.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable profiling by default.")
    wait_steps: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Vector steps skipped before warmup; null aligns capture with the "
            "first PPO update."
        ),
    )
    warmup_steps: NonNegativeInt = Field(
        default=1, description="Vector steps used for profiler warmup."
    )
    active_steps: PositiveInt = Field(
        default=2, description="Vector steps recorded in each profile window."
    )
    repeat: PositiveInt = Field(
        default=1, description="Number of profile windows to record."
    )
    output_dir: str = Field(
        default="./logs/profiles",
        min_length=1,
        description="Profiler trace output directory.",
    )
    record_shapes: bool = Field(
        default=True, description="Record tensor shapes for operators."
    )
    profile_memory: bool = Field(
        default=True, description="Record tensor memory allocations."
    )
    with_stack: bool = Field(
        default=False,
        description="Record Python stacks; adds trace overhead and size.",
    )

    def resolve_schedule(self, rollout_size: int) -> tuple[int, int]:
        """Resolve wait and warmup lengths for the configured rollout size.

        Args:
            rollout_size: Number of vector steps in one rollout buffer.

        Returns:
            Pair of ``(wait_steps, warmup_steps)`` for ``torch.profiler``.
            With automatic alignment, the first active profiler step is the
            iteration that fills the first rollout buffer.

        Raises:
            ValueError: If ``rollout_size`` is not positive.
        """
        if rollout_size < 1:
            raise ValueError("rollout_size must be positive")
        if self.wait_steps is not None:
            return self.wait_steps, self.warmup_steps

        warmup_steps = min(self.warmup_steps, rollout_size - 1)
        wait_steps = rollout_size - 1 - warmup_steps
        return wait_steps, warmup_steps


def load_profiling_config(path: str | Path) -> ProfilingConfig:
    """Load and validate profiling settings from a YAML file.

    Args:
        path: Path to a YAML mapping containing profiling settings.

    Returns:
        Validated profiling settings.

    Raises:
        ValueError: If the file cannot be read, contains invalid YAML, or does
            not contain a YAML mapping.
    """
    config_path = Path(path)
    try:
        with config_path.open("rt", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read profiling config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid profiling YAML in {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Profiling config {config_path} must contain a YAML mapping")
    return ProfilingConfig.model_validate(raw)
