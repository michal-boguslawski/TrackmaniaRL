"""Collector callback: logs run configuration parameters at rollout start.

Logs the collector config (merged buffer/trainer/network settings) and
optionally the full run config dict to all metrics loggers.
"""

from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import Callback


class ParamsLoggingCallback(Callback):
    """Log training configuration parameters at rollout start.

    On rollout_start, logs:
    - Collector config dict (from RolloutCollector.config())
    - Full run config dict (if provided at construction)
    """

    def __init__(self, logger: MetricsLogger, run_config: dict | None = None):
        """Initialize parameter logging callback.

        Args:
            logger: MetricsLogger backend.
            run_config: Full run config dict for MLflow parameter logging.
        """
        self._logger = logger
        self._run_config = run_config

    def on_rollout_start(self, config: dict | None = None, *args, **kwargs):
        """Log collector config and full run config if available."""
        if config:
            self._logger.log_parameters(config)
            if self._run_config is not None:
                self._logger.log_config(self._run_config)

    def on_env_step(self, *args, **kwargs):
        pass

    def on_rollout_end(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass
