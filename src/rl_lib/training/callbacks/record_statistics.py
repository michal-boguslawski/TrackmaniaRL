"""Collector callback: logs episode statistics from vector environment info.

Reads episode returns/lengths from Gymnasium's RecordEpisodeStatistics wrapper
output in the info dict. Supports two modes:
- step: log mean return/length per step (per batch of done episodes)
- mean: accumulate and log running mean at flush (end of rollout)
"""

from typing import Any

from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import Callback
from rl_lib.run_config import EpisodeStatisticsCallbackSettings


class RecordStatisticLoggerCallback(Callback):
    """Log episode statistics from vector environment info dict.

    Extracts episode returns and lengths from the stats_key entry in info
    (populated by RecordEpisodeStatistics wrapper). In 'step' mode, logs
    per-step batch means; in 'mean' mode, accumulates and logs running
    average at flush time.

    Attributes:
        _logger: MetricsLogger backend.
        cfg: EpisodeStatisticsCallbackSettings (mode, stats_key).
        _returns_sum: Accumulated sum of episode returns (mean mode).
        _lengths_sum: Accumulated sum of episode lengths (mean mode).
        _n: Total episodes accumulated (mean mode).
        _last_step: Step of last episode batch (mean mode).
    """

    def __init__(
        self,
        logger: MetricsLogger,
        config: EpisodeStatisticsCallbackSettings,
    ):
        """Initialize statistics logging callback.

        Args:
            logger: MetricsLogger backend.
            config: Settings with mode ('step' or 'mean') and stats_key.
        """
        self._logger = logger
        self.cfg = config
        self._returns_sum = 0.0
        self._lengths_sum = 0.0
        self._n = 0
        self._last_step: int | None = None

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, info: dict[str, Any], *args, **kwargs):
        """Process completed episodes from info dict."""
        if self.cfg.stats_key in info:
            dones = info[f"_{self.cfg.stats_key}"]
            returns = info[self.cfg.stats_key]["r"][dones]
            lengths = info[self.cfg.stats_key]["l"][dones]
            n = int(dones.sum())

            if self.cfg.mode == "step":
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
        """Log accumulated mean statistics (for 'mean' mode)."""
        if self.cfg.mode != "mean" or self._n == 0:
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
