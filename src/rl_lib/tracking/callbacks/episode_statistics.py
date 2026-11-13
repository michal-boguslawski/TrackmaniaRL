"""Buffer and log completed-episode statistics from vector-environment info."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.config.callbacks.episode_statistics import (
    EpisodeStatisticsCallbackSettings,
)
from rl_lib.tracking.sinks.base import MetricsSink


class EpisodeStatisticLoggerCallback(Callback):
    """Log completed-episode returns and lengths from vector-environment info.

    The buffer contains one summary for each environment step that completed at
    least one episode. ``step`` mode writes those summaries separately, while
    ``mean`` mode writes episode-weighted means for the whole buffer. A full
    buffer is flushed automatically when it reaches the configured length.
    Calling ``flush`` explicitly also drains a partial buffer.

    Attributes:
        cfg: Settings for the statistics key, aggregation mode, and buffer size.
        _logger: Metrics sink receiving episode metrics.
        _buffer: Ordered ``(metrics, step)`` summaries awaiting a flush.
    """

    def __init__(
        self,
        logger: MetricsSink,
        config: EpisodeStatisticsCallbackSettings,
    ) -> None:
        """Initialize the episode-statistics callback.

        Args:
            logger: Metrics sink that receives the computed statistics.
            config: Statistics mode, info key, and maximum buffered summaries.
        """
        self._logger = logger
        self.cfg = config
        self._buffer: list[tuple[dict[str, float], int]] = []

    def on_event(
        self,
        event: CallbackEvent,
        step: int | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle environment-step events only.

        Args:
            event: Event being dispatched.
            **kwargs: Environment step payload (``step`` and ``info``) for
                ``ENV_STEP`` events.
        """
        if event == CallbackEvent.ENV_STEP:
            self._handle_env_step(step, info)

    def _handle_env_step(self, step: int | None, info: dict[str, Any]) -> None:
        """Extract, buffer, and flush episode metrics for one environment step."""
        if step is None or not isinstance(info, Mapping):
            return

        batch = self._episode_batch(info)
        if batch is None:
            return

        returns, lengths = batch
        episode_count = int(returns.size)
        metrics = {
            "episode/returns": float(returns.mean()),
            "episode/lengths": float(lengths.mean()),
            "episode/n": float(episode_count),
        }
        self._buffer.append((metrics, int(step)))
        if len(self._buffer) >= self.cfg.buffer_length:
            self.flush()

    def _episode_batch(
        self, info: Mapping[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Read completed episode return/length arrays from vector info."""
        episode_info = info.get("final_info")
        if (
            not isinstance(episode_info, Mapping)
            or self.cfg.stats_key not in episode_info
        ):
            episode_info = info

        mask_key = f"_{self.cfg.stats_key}"
        if self.cfg.stats_key in episode_info and mask_key in episode_info:
            dones = episode_info[mask_key]
            returns = episode_info[self.cfg.stats_key]["r"][dones]
            lengths = episode_info[self.cfg.stats_key]["l"][dones]
            return returns, lengths

    def flush(self) -> None:
        """Write buffered episode metrics and clear the buffer."""
        if not self._buffer:
            return

        if self.cfg.mode == "step":
            for metrics, step in self._buffer:
                self._logger.log_metrics(metrics, step=step)
        else:
            episode_count = sum(
                int(metrics["episode/n"]) for metrics, _ in self._buffer
            )
            metrics = {
                "episode/returns": sum(
                    entry["episode/returns"] * entry["episode/n"]
                    for entry, _ in self._buffer
                )
                / episode_count,
                "episode/lengths": sum(
                    entry["episode/lengths"] * entry["episode/n"]
                    for entry, _ in self._buffer
                )
                / episode_count,
                "episode/n": float(episode_count),
            }
            self._logger.log_metrics(
                metrics, step=max(step for _, step in self._buffer)
            )
        self._buffer.clear()
