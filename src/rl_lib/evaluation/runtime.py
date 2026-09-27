"""Evaluation runtime helpers: stats key extraction and default loggers."""

from __future__ import annotations

from rl_lib.run_config import EnvironmentSettings
from rl_lib.tracking.base import MetricsLogger


def _episode_stats_key(environment: EnvironmentSettings) -> str:
    """Get the episode stats key from the environment wrapper config.

    Searches the wrapper list for record_episode_stats and returns
    its stats_key, defaulting to "episode".

    Args:
        environment: EnvironmentSettings with wrapper configurations.

    Returns:
        Episode statistics key string.
    """
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
    """Return default metrics loggers if none provided.

    Defaults to ConsoleMetricsLogger for local development.

    Args:
        metrics_loggers: Explicit logger list or None.

    Returns:
        List of MetricsLogger instances.
    """
    if metrics_loggers is not None:
        return metrics_loggers
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    return [ConsoleMetricsLogger()]
