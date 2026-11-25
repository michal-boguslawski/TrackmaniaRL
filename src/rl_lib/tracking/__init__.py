"""Experiment tracking package.

Provides MetricsLogger interface and implementations:
- MetricsLogger: Abstract base class (base.py)
- MLflowLogger: MLflow backend with context manager (mlflow_logger.py)
- ConsoleMetricsLogger: Console logging for development (console_logger.py)
- LocalArtifactLogger: Local model and state-dictionary artifacts
"""

from rl_lib.tracking.sinks.base import MetricsSink
from rl_lib.tracking.sinks.local_sink import LocalMetricsSink
from rl_lib.tracking.sinks.mlflow_sink import (
    MLflowMetricsSink,
    run_status_for_exception,
)

__all__ = [
    "LocalMetricsSink",
    "MLflowMetricsSink",
    "MetricsSink",
    "run_status_for_exception",
]
