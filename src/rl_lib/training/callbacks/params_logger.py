# rl_lib/training/callbacks/params_logger.py
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback


class ParamsLoggingCallback(CollectorCallback):
    def __init__(self, logger: MetricsLogger, run_config: dict | None = None):
        self._logger = logger
        self._run_config = run_config

    def on_rollout_start(self, config: dict | None = None, *args, **kwargs):
        if config:
            self._logger.log_parameters(config)
            if self._run_config is not None and hasattr(self._logger, "log_config"):
                self._logger.log_config(self._run_config)

    def on_env_step(self, *args, **kwargs):
        pass

    def on_rollout_end(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass
