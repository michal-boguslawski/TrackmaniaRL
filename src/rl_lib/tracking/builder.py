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


def _build_mlflow_metrics_sink(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args,
    **kwargs,
) -> MLflowMetricsSink:
    return MLflowMetricsSink(
        experiment_name=run_paths.experiment,
        run_name=run_paths.session_id,
        tags={"algorithm": run_paths.algorithm} if run_paths.algorithm else None,
        log_system_metrics=tracking_settings.log_system_metrics,
    )


def _build_console_metrics_sink(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args,
    **kwargs,
) -> LocalMetricsSink:
    return LocalMetricsSink(run_paths=run_paths)


def build_metrics_logger(
    tracking_settings: TrackingSettings,
    run_paths: RunPaths,
    *args,
    **kwargs,
) -> CompositeMetricsLogger:
    logger = CompositeMetricsLogger()
    if tracking_settings.mlflow:
        logger.add_sink(
            _build_mlflow_metrics_sink(tracking_settings, run_paths=run_paths)
        )

    if tracking_settings.console:
        logger.add_sink(_build_console_metrics_sink(tracking_settings, run_paths))
    return logger


def build_callbacks(
    metrics_logger: MetricsSink,
    callback_settings_list: list[CallbackSettings],
    base_env: EnvironmentSettings,
    run_paths: RunPaths,
    device: T.device | str = "cpu",
    run_seed: int | None = None,
    modules: nn.Module | None = None,
) -> CallbackList | None:
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
