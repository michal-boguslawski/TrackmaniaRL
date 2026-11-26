"""Tests for buffered episode-statistics callback behavior."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_lib.tracking.callbacks.base import CallbackEvent
from rl_lib.tracking.callbacks.record_statistics import RecordStatisticLoggerCallback
from rl_lib.tracking.config.callbacks.episode_statistics import (
    EpisodeStatisticsCallbackSettings,
)
from rl_lib.tracking.sinks.base import MetricsSink


class _MemorySink(MetricsSink):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((dict(metrics), step))


def _episode_info(returns: list[float], lengths: list[float]) -> dict[str, Any]:
    return {
        "_episode": np.ones(len(returns), dtype=bool),
        "episode": {
            "r": np.asarray(returns, dtype=np.float64),
            "l": np.asarray(lengths, dtype=np.float64),
        },
    }


def test_on_event_processes_only_environment_steps() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(mode="step", buffer_length=1),
    )

    callback.on_event(
        CallbackEvent.ROLLOUT_START,
        step=2,
        info=_episode_info([3.0], [12.0]),
    )

    assert sink.metrics == []
    assert callback._buffer == []


def test_step_mode_flushes_each_buffered_step_separately() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(
            mode="step", buffer_length=2
        ),
    )

    callback.on_event(
        CallbackEvent.ENV_STEP, step=3, info=_episode_info([2.0, 4.0], [10, 20])
    )
    assert sink.metrics == []

    callback.on_event(
        CallbackEvent.ENV_STEP, step=4, info=_episode_info([6.0], [30])
    )

    assert sink.metrics == [
        (
            {"episode/returns": 3.0, "episode/lengths": 15.0, "episode/n": 2.0},
            3,
        ),
        (
            {"episode/returns": 6.0, "episode/lengths": 30.0, "episode/n": 1.0},
            4,
        ),
    ]


def test_mean_mode_logs_episode_weighted_metrics_at_latest_step() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(mode="mean", buffer_length=2),
    )

    callback.on_event(
        CallbackEvent.ENV_STEP, step=3, info=_episode_info([2.0, 4.0], [10, 20])
    )
    callback.on_event(
        CallbackEvent.ENV_STEP, step=8, info=_episode_info([6.0], [30])
    )

    assert sink.metrics == [
        (
            {"episode/returns": 4.0, "episode/lengths": 20.0, "episode/n": 3.0},
            8,
        )
    ]
    assert callback._buffer == []


def test_flush_drains_a_partial_buffer() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(mode="mean", buffer_length=4),
    )
    callback.on_event(
        CallbackEvent.ENV_STEP, step=5, info=_episode_info([7.0], [14.0])
    )

    callback.flush()

    assert sink.metrics == [
        ({"episode/returns": 7.0, "episode/lengths": 14.0, "episode/n": 1.0}, 5)
    ]


def test_nested_final_info_statistics_are_read() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(mode="step", buffer_length=1),
    )

    callback.on_event(
        CallbackEvent.ENV_STEP,
        step=6,
        info={
            "final_info": {
                "_episode": np.array([True, False]),
                "episode": {"r": np.array([3.0, 0.0]), "l": np.array([12, 0])},
            },
        },
    )

    assert sink.metrics == [
        ({"episode/returns": 3.0, "episode/lengths": 12.0, "episode/n": 1.0}, 6)
    ]


def test_per_environment_final_info_statistics_are_read() -> None:
    sink = _MemorySink()
    callback = RecordStatisticLoggerCallback(
        sink,
        EpisodeStatisticsCallbackSettings(mode="step", buffer_length=1),
    )

    callback.on_event(
        CallbackEvent.ENV_STEP,
        step=7,
        info={
            "_final_info": np.array([True, False]),
            "final_info": [
                {"episode": {"r": 5.0, "l": 9}},
                None,
            ],
        },
    )

    assert sink.metrics == [
        ({"episode/returns": 5.0, "episode/lengths": 9.0, "episode/n": 1.0}, 7)
    ]
