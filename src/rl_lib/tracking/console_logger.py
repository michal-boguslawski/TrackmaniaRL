from logging import getLogger

from rl_lib.tracking.base import MetricsLogger


logger = getLogger(__name__)

class ConsoleMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        formatted = {k: f"{v:.6f}" for k, v in metrics.items()}
        logger.debug(f"Step {step}: {formatted}")

    def log_parameters(self, parameters: dict[str, int | float | str]) -> None:
        formatted = {k: f"{v}" for k, v in parameters.items()}
        logger.debug(formatted)
