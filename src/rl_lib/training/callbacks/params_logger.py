# rl_lib/training/callbacks/params_logger.py
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback


class ParamsLoggingCallback(CollectorCallback):
    def __init__(self, logger: MetricsLogger):
        self._logger = logger

    def on_rollout_start(self, config: dict | None = None, *args, **kwargs):
        if config:
            self._logger.log_parameters(config)

    def on_env_step(self, *args, **kwargs):
        pass

    def on_rollout_end(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass
