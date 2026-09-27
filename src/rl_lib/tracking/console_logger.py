"""Console metrics logger for development/debugging.

Logs metrics and parameters to Python logging at DEBUG level.
Useful for local development without MLflow server.
"""

from logging import getLogger

from rl_lib.tracking.base import MetricsLogger


logger = getLogger(__name__)

class ConsoleMetricsLogger(MetricsLogger):
    """Simple console logger for metrics and parameters.

    Formats metrics with 6 decimal places and logs at DEBUG level.
    Parameters are logged as-is.
    """

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to console at DEBUG level."""
        formatted = {k: f"{v:.6f}" for k, v in metrics.items()}
        logger.debug(f"Step {step}: {formatted}")

    def log_parameters(self, parameters: dict[str, int | float | str]) -> None:
        """Log parameters to console at DEBUG level."""
        formatted = {k: f"{v}" for k, v in parameters.items()}
        logger.debug(formatted)
