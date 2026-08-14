from logging import getLogger

from src.rl_lib.tracking.base import MetricsLogger


logger = getLogger(__name__)

class ConsoleMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        formatted = {k: f"{v:.4f}" for k, v in metrics.items()}
        logger.debug(f"Step {step}: {formatted}")
