import torch as T

from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.paths import RunPaths
from rl_lib.tracking.callbacks.base import CallbackList
from rl_lib.tracking.callbacks.registry import CALLBACK_REGISTRY
from rl_lib.tracking.composite_logger import CompositeMetricsLogger
from rl_lib.tracking.config.callbacks import CallbackSettings
from rl_lib.tracking.config.metrics_logger import TrackingSettings
from rl_lib.tracking.sinks.base import MetricsSink
from rl_lib.tracking.sinks.console_sink import ConsoleMetricsSink
from rl_lib.tracking.sinks.mlflow_sink import MLflowMetricsSink


def _build_mlflow_metrics_sink(
    tracking_settings: TrackingSettings,
    env_id: str,
    session_id: str,
    algorithm: str | None = None,
    *args,
    **kwargs,
) -> MLflowMetricsSink:
    return MLflowMetricsSink(
        experiment_name=tracking_settings.experiment_name or env_id,
        run_name=tracking_settings.run_name or session_id,
        tags={"algorithm": algorithm} if algorithm else None,
        log_system_metrics=tracking_settings.log_system_metrics,
    )


def _build_console_metrics_sink(
    tracking_settings: TrackingSettings,
    *args,
    **kwargs,
) -> ConsoleMetricsSink:
    return ConsoleMetricsSink()


def build_metrics_logger(
    tracking_settings: TrackingSettings,
    env_id: str,
    session_id: str,
    algorithm: str | None = None,
    *args,
    **kwargs,
) -> CompositeMetricsLogger:
    logger = CompositeMetricsLogger()
    if tracking_settings.mlflow:
        logger.add_sink(
            _build_mlflow_metrics_sink(
                tracking_settings,
                env_id=env_id,
                session_id=session_id,
                algorithm=algorithm,
            )
        )

    if tracking_settings.console:
        logger.add_sink(_build_console_metrics_sink(tracking_settings))
    return logger


def build_callbacks(
    metrics_logger: MetricsSink,
    callback_settings_list: list[CallbackSettings],
    base_env: EnvironmentSettings,
    run_paths: RunPaths,
    device: T.device | str = "cpu",
    run_seed: int | None = None,
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
            )
            for callback_settings in callback_settings_list
        ]
    )
