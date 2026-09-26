from typing import Any, Literal

from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback
from rl_lib.run_config import CallbackSettings


class RecordStatisticLoggerCallback(CollectorCallback):
    def __init__(
        self,
        logger: MetricsLogger,
        mode: Literal["step", "mean"] = "step",
        stats_key: str | None = None,
    ):
        self._logger = logger
        self.mode = mode
        self.stats_key = stats_key or CallbackSettings().episode_statistics_key
        self._returns_sum = 0.0
        self._lengths_sum = 0.0
        self._n = 0
        self._last_step: int | None = None

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, info: dict[str, Any], *args, **kwargs):
        if self.stats_key in info:
            dones = info[f"_{self.stats_key}"]
            returns = info[self.stats_key]["r"][dones]
            lengths = info[self.stats_key]["l"][dones]
            n = int(dones.sum())

            if self.mode == "step":
                metrics = {
                    "episode/returns": returns.mean(),
                    "episode/lengths": lengths.mean(),
                    "episode/n": n,
                }
                self._logger.log_metrics(metrics, step=step)
            else:
                self._returns_sum += float(returns.sum())
                self._lengths_sum += float(lengths.sum())
                self._n += n
                self._last_step = step

    def on_rollout_end(self, *args, **kwargs):
        self.flush()

    def flush(self):
        if self.mode != "mean" or self._n == 0:
            return

        metrics = {
            "episode/returns": self._returns_sum / self._n,
            "episode/lengths": self._lengths_sum / self._n,
            "episode/n": self._n,
        }
        self._logger.log_metrics(metrics, step=self._last_step)

        self._returns_sum = 0.0
        self._lengths_sum = 0.0
        self._n = 0
        self._last_step = None
