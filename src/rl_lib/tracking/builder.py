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
) -> MLflowMetricsSink:
    return MLflowMetricsSink(
        experiment_name=tracking_settings.experiment_name,
        run_name=tracking_settings.run_name,
        log_system_metrics=tracking_settings.log_system_metrics,
    )


def _build_console_metrics_sink(
    tracking_settings: TrackingSettings,
) -> ConsoleMetricsSink:
    return ConsoleMetricsSink()


def build_metrics_logger(
    tracking_settings: TrackingSettings, session_id: str
) -> CompositeMetricsLogger:
    metrics_logger = CompositeMetricsLogger()
    if tracking_settings.mlflow:
        metrics_logger.add_sink(_build_mlflow_metrics_sink(tracking_settings))

    if tracking_settings.console:
        metrics_logger.add_sink(_build_console_metrics_sink(tracking_settings))
    return metrics_logger


def build_callbacks(
    metrics_logger: MetricsSink,
    callback_settings_list: list[CallbackSettings],
) -> CallbackList | None:
    if not callback_settings_list:
        return
    return CallbackList(
        [
            CALLBACK_REGISTRY[callback_settings.name](metrics_logger, callback_settings)
            for callback_settings in callback_settings_list
        ]
    )
