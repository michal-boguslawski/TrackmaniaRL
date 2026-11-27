"""Construct tracking sinks and training callbacks from run configuration."""

from types import TracebackType
from typing import Self

import torch as T
from torch import nn

from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.paths import RunPaths
from rl_lib.tracking.callbacks.base import CallbackList
from rl_lib.tracking.callbacks.registry import CALLBACK_REGISTRY
from rl_lib.tracking.composite_logger import CompositeMetricsLogger
from rl_lib.tracking.config.callbacks import CallbackSettings
from rl_lib.tracking.config.metrics_logger import TrackingSettings
from rl_lib.tracking.sinks.base import MetricsSink
from rl_lib.tracking.sinks.local_sink import LocalMetricsSink
from rl_lib.tracking.sinks.mlflow_sink import MLflowMetricsSink
from rl_lib.tracking.utils import run_status_for_exception


class _MetricsLoggerBuildGuard:
    """Close partially constructed tracking sinks if logger setup fails."""

    def __init__(self, metrics_logger: CompositeMetricsLogger) -> None:
        """Initialize the guard for a composite logger."""
        self._metrics_logger = metrics_logger
        self._committed = False

    def __enter__(self) -> Self:
        """Enter the guarded logger-construction scope."""
        return self

    def commit(self) -> None:
        """Transfer cleanup ownership to the caller."""
        self._committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close partial sinks on failure and preserve exception propagation."""
        if not self._committed:
            self._metrics_logger.close(
                status=run_status_for_exception(exc_type, exc_val)
            )
        return False


def _build_mlflow_metrics_sink(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args: object,
    **kwargs: object,
) -> MLflowMetricsSink:
    """Create the MLflow sink for this run."""
    return MLflowMetricsSink(
        experiment_name=run_paths.experiment,
        run_name=run_paths.session_id,
        tags={"algorithm": run_paths.algorithm} if run_paths.algorithm else None,
        log_system_metrics=tracking_settings.log_system_metrics,
    )


def _build_console_metrics_sink(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args: object,
    **kwargs: object,
) -> LocalMetricsSink:
    """Create the local console and artifact sink for this run."""
    return LocalMetricsSink(run_paths=run_paths)


def build_metrics_logger(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args: object,
    **kwargs: object,
) -> CompositeMetricsLogger:
    """Build configured sinks and clean up if construction is interrupted.

    Args:
        tracking_settings: Settings selecting and configuring tracking sinks.
        run_paths: Paths and run identity shared by the sinks.
        *args: Ignored compatibility arguments.
        **kwargs: Ignored compatibility keyword arguments.

    Returns:
        Composite logger that owns all configured sinks.
    """
    metrics_logger = CompositeMetricsLogger()
    with _MetricsLoggerBuildGuard(metrics_logger) as guard:
        if tracking_settings.mlflow:
            metrics_logger.add_sink(
                _build_mlflow_metrics_sink(tracking_settings, run_paths=run_paths)
            )

        if tracking_settings.console:
            metrics_logger.add_sink(
                _build_console_metrics_sink(tracking_settings, run_paths)
            )
        guard.commit()
    return metrics_logger


def build_callbacks(
    metrics_logger: MetricsSink,
    callback_settings_list: list[CallbackSettings],
    base_env: EnvironmentSettings,
    run_paths: RunPaths,
    device: T.device | str = "cpu",
    run_seed: int | None = None,
    modules: nn.Module | None = None,
) -> CallbackList | None:
    """Build enabled callbacks in their configured order.

    Args:
        metrics_logger: Logger shared with all callbacks.
        callback_settings_list: Callback settings to instantiate.
        base_env: Environment configuration used by environment-owning callbacks.
        run_paths: Paths and identity for this training run.
        device: Device used by evaluation or recording callbacks.
        run_seed: Seed assigned to callback-owned environments.
        modules: Model modules used by checkpoint and evaluation callbacks.

    Returns:
        Ordered callback list, or ``None`` when no callbacks are enabled.
    """
    if not callback_settings_list:
        return
    return CallbackList(
        [
            CALLBACK_REGISTRY[callback_settings.name](
                logger=metrics_logger,
                config=callback_settings,
                base_env=base_env,
                run_paths=run_paths,
                device=device,
                run_seed=run_seed,
                modules=modules,
            )
            for callback_settings in callback_settings_list
        ]
    )
