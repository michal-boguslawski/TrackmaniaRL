"""Experiment tracking package.

Provides MetricsLogger interface and implementations:
- MetricsLogger: Abstract base class (base.py)
- MLflowLogger: MLflow backend with context manager (mlflow_logger.py)
- ConsoleMetricsLogger: Console logging for development (console_logger.py)
- LocalArtifactLogger: Local model and state-dictionary artifacts
"""

from rl_lib.tracking.base import MetricsLogger
from rl_lib.tracking.mlflow_logger import MLflowLogger, run_status_for_exception
from rl_lib.tracking.console_logger import ConsoleMetricsLogger
from rl_lib.tracking.local_artifact_logger import LocalArtifactLogger

__all__ = [
    "MetricsLogger",
    "MLflowLogger",
    "run_status_for_exception",
    "ConsoleMetricsLogger",
    "LocalArtifactLogger",
]
