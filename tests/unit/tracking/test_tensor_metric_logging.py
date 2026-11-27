"""Tests for deferred tensor-to-scalar conversion in metrics logging."""

import pytest
import torch as T

import rl_lib.tracking.callbacks.metrics_logger as metrics_logger_module
from rl_lib.tracking.callbacks.base import CallbackEvent
from rl_lib.tracking.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.tracking.config.callbacks.metrics_logger import MetricsCallbackSettings
from rl_lib.tracking.metric_values import metric_values_to_scalars


class _MemorySink:
    """Collect logged scalar metrics for assertions."""

    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))


def test_metric_values_to_scalars_transfers_scalar_tensors() -> None:
    value = T.tensor(3.5)

    result = metric_values_to_scalars(
        {"tensor": value, "other_tensor": T.tensor(2.0)}
    )

    assert result == {"tensor": 3.5, "other_tensor": 2.0}


def test_metric_values_to_scalars_rejects_non_scalar_tensors() -> None:
    with pytest.raises(ValueError, match="zero-dimensional"):
        metric_values_to_scalars({"vector": T.tensor([1.0, 2.0])})


def test_metric_values_to_scalars_rejects_grad_tracked_metrics() -> None:
    with pytest.raises(ValueError, match="detached"):
        metric_values_to_scalars({"loss": T.tensor(1.0, requires_grad=True)})


def test_epoch_metrics_are_averaged_before_the_single_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _MemorySink()
    callback = MetricsLoggingCallback(
        sink, MetricsCallbackSettings(granularity=CallbackEvent.EPOCH_END)
    )
    conversions: list[dict[str, object]] = []
    original = metrics_logger_module.metric_values_to_scalars

    def counted(metrics):
        conversions.append(dict(metrics))
        return original(metrics)

    monkeypatch.setattr(metrics_logger_module, "metric_values_to_scalars", counted)
    callback.on_event(
        CallbackEvent.MINIBATCH_END,
        metrics={
            "loss/total": T.tensor(1.0),
            "optimization/lr": T.tensor(0.001),
        },
    )
    callback.on_event(
        CallbackEvent.MINIBATCH_END,
        metrics={
            "loss/total": T.tensor(3.0),
            "optimization/lr": T.tensor(0.003),
        },
    )

    assert conversions == []
    callback.on_event(CallbackEvent.EPOCH_END, step=12)

    assert len(conversions) == 1
    assert sink.metrics == []
    callback.flush()
    assert sink.metrics == [
        (
            {
                "train/loss/total": pytest.approx(2.0),
                "train/optimization/lr": pytest.approx(0.002),
            },
            12,
        )
    ]


def test_logging_callback_does_not_filter_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _MemorySink()
    callback = MetricsLoggingCallback(
        sink,
        MetricsCallbackSettings(granularity=CallbackEvent.MINIBATCH_END),
    )
    converted_names: list[set[str]] = []
    original = metrics_logger_module.metric_values_to_scalars

    def record_names(metrics):
        converted_names.append(set(metrics))
        return original(metrics)

    monkeypatch.setattr(metrics_logger_module, "metric_values_to_scalars", record_names)
    callback.on_event(
        CallbackEvent.MINIBATCH_END,
        step=3,
        metrics={
            "total": T.tensor(1.0),
            "diagnostic/entropy": T.tensor(0.5),
        },
    )

    assert converted_names == [{"total", "diagnostic/entropy"}]
    callback.flush()
    assert sink.metrics == [
        (
            {"train/total": 1.0, "train/diagnostic/entropy": 0.5},
            3,
        )
    ]
