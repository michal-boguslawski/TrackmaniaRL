from __future__ import annotations

from rl_lib.run_config import EnvironmentSettings
from rl_lib.tracking.base import MetricsLogger


def _episode_stats_key(environment: EnvironmentSettings) -> str:
    """Get the episode stats key from the environment wrapper config."""
    return next(
        (
            wrapper.stats_key
            for wrapper in environment.wrappers
            if wrapper.name == "record_episode_stats"
        ),
        "episode",
    )


def _default_metrics_loggers(
    metrics_loggers: list[MetricsLogger] | None,
) -> list[MetricsLogger]:
    """Return default metrics loggers if none provided."""
    if metrics_loggers is not None:
        return metrics_loggers
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    return [ConsoleMetricsLogger()]
