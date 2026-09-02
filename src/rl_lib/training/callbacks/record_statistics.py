from typing import Any, Literal

from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback


class RecordStatisticLoggerCallback(CollectorCallback):
    def __init__(self, logger: MetricsLogger, mode: Literal["step", "mean"] = "step"):
        self._logger = logger
        self.mode = mode
        self._returns_sum = 0.0
        self._lengths_sum = 0.0
        self._n = 0
        self._last_step: int | None = None

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, info: dict[str, Any], *args, **kwargs):
        if "episode" in info:
            dones = info["_episode"]
            returns = info["episode"]["r"][dones]
            lengths = info["episode"]["l"][dones]
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
