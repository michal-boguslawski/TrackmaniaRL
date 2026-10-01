"""Metric verbosity shared by the trainer and the tracking backends.

Verbosity answers two questions with one setting: which metrics are worth
computing, and which of those are worth logging.

- ``0``: no scalar metrics at all.
- ``1``: only the core training losses (total, actor, critic, entropy).
- ``2``: every metric.

The components that produce metrics ask the verbosity before they build the
diagnostic part of a metric dict, so a lower verbosity removes the tensor
reductions and the host synchronizations rather than just hiding the result.
``VerbosityMetricsLogger`` then filters what reaches each backend, which also
covers producers that report metrics the trainer never computed.

Only ``log_metrics`` is filtered. Parameters, configs, models, and artifact
files are run metadata rather than metrics, so they are always written.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Iterable

from rl_lib.tracking.base import MetricsLogger


class Verbosity(IntEnum):
    """How much of the metric stream is computed and logged.

    Attributes:
        NONE: Log nothing; only metrics needed for training control flow run.
        CORE: Log the core training losses (total, actor, critic, entropy).
        ALL: Log every metric, including diagnostics.
    """

    NONE = 0
    CORE = 1
    ALL = 2

    @property
    def logs_core_losses(self) -> bool:
        """Whether the core training losses are produced and logged."""
        return self >= Verbosity.CORE

    @property
    def logs_diagnostics(self) -> bool:
        """Whether the diagnostic metrics beyond the core losses are logged."""
        return self >= Verbosity.ALL


def as_verbosity(verbosity: int | Verbosity) -> Verbosity:
    """Coerce a verbosity to its enum member.

    Args:
        verbosity: Verbosity level, as an int or a ``Verbosity`` member.

    Returns:
        The matching ``Verbosity`` member.

    Raises:
        ValueError: If verbosity is not one of 0, 1, or 2.
    """
    if isinstance(verbosity, Verbosity):
        return verbosity
    try:
        return Verbosity(verbosity)
    except ValueError as error:
        levels = [int(level) for level in Verbosity]
        raise ValueError(
            f"verbosity must be one of {levels}, got {verbosity!r}"
        ) from error


# Losses are the ``loss/`` family; the actor surrogate loss is the one loss
# reported under the ``metrics/`` prefix, so it is named explicitly.
CORE_LOSS_PREFIXES: tuple[str, ...] = ("loss/",)
CORE_LOSS_METRICS: frozenset[str] = frozenset({"metrics/actor_loss"})



def is_core_loss(name: str) -> bool:
    """Return whether a metric name is one of the core training losses.

    Args:
        name: Metric name as produced by the trainer (e.g. ``loss/critic``).

    Returns:
        True for total, actor, critic, and entropy losses.
    """
    return name.startswith(CORE_LOSS_PREFIXES) or name in CORE_LOSS_METRICS


def filter_metrics(
    metrics: dict[str, float], verbosity: int | Verbosity
) -> dict[str, float]:
    """Select the metrics allowed at a verbosity level.

    Args:
        metrics: Metric name -> value mapping as produced by a callback.
        verbosity: Metric detail level (0, 1, or 2).

    Returns:
        The subset of ``metrics`` to log. The input is returned unchanged at
        verbosity 2 and as an empty dict at verbosity 0.

    Raises:
        ValueError: If verbosity is not one of 0, 1, or 2.
    """
    level = as_verbosity(verbosity)
    if level == Verbosity.ALL:
        return dict(metrics)
    if level == Verbosity.NONE:
        return {}
    return {name: value for name, value in metrics.items() if is_core_loss(name)}


def filter_loggers(
    loggers: Iterable[MetricsLogger], verbosity: int | Verbosity
) -> list[MetricsLogger]:
    """Wrap loggers so each one only receives metrics allowed at ``verbosity``.

    Loggers that already filter are left alone, so composing this with an
    explicit filter is harmless.

    Args:
        loggers: Tracking backends to wrap.
        verbosity: Metric detail level (0, 1, or 2).

    Returns:
        List of loggers, each wrapped in a ``VerbosityMetricsLogger``.
    """
    level = as_verbosity(verbosity)
    return [
        logger
        if isinstance(logger, VerbosityMetricsLogger) or level == Verbosity.ALL
        else VerbosityMetricsLogger(logger, level)
        for logger in loggers
    ]


class VerbosityMetricsLogger(MetricsLogger):
    """MetricsLogger decorator that drops metrics above the verbosity budget.

    Every method other than ``log_metrics`` is delegated untouched, so
    artifacts, parameters, and configs keep being written at any verbosity.
    Empty metric dicts are never forwarded, so a fully filtered flush costs
    no backend round trip.

    Attributes:
        logger: Wrapped backend receiving the surviving metrics.
        verbosity: Resolved verbosity level.
    """

    def __init__(self, logger: MetricsLogger, verbosity: int | Verbosity):
        """Initialize the decorator.

        Args:
            logger: Backend to forward surviving metrics to.
            verbosity: Metric detail level (0, 1, or 2).

        Raises:
            ValueError: If verbosity is not one of 0, 1, or 2.
        """
        self.logger = logger
        self.verbosity = as_verbosity(verbosity)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Forward only the metrics allowed at this verbosity."""
        selected = filter_metrics(metrics, self.verbosity)
        if selected:
            self.logger.log_metrics(selected, step)

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Delegate parameter logging; parameters are not metrics."""
        self.logger.log_parameters(parameters)

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        """Delegate config artifact logging; configs are not metrics."""
        self.logger.log_config(config, artifact_file)

    def log_evaluation(
        self,
        episodes: list[dict[str, float | int]],
        summary: dict[str, float],
        step: int,
        scope: str,
        config_artifact: str = "config/run_config.yaml",
    ) -> None:
        """Delegate evaluation persistence; evaluation files are not metrics."""
        self.logger.log_evaluation(episodes, summary, step, scope, config_artifact)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Delegate model artifact logging."""
        self.logger.log_model(model, artifact_path, registered_model_name)

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        """Delegate state-dict artifact logging."""
        self.logger.log_state_dict(state_dict, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Delegate file artifact logging."""
        self.logger.log_artifact(local_path, artifact_path)
