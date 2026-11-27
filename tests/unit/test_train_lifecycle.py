"""Tests for training-run cleanup and final tracking status."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rl_lib.train as train_module
import rl_lib.tracking.builder as tracking_builder
from rl_lib.tracking.composite_logger import CompositeMetricsLogger
from rl_lib.tracking.utils import run_status_for_exception


class _FakeMetricsLogger:
    def __init__(self, events):
        self.events = events

    def log_config(self, config):
        pass

    def log_parameters(self, parameters):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(status=run_status_for_exception(exc_type, exc_val))
        return False

    def close(self, status="FINISHED"):
        self.events.append(("metrics.close", status))


class _FakeModules:
    def config(self):
        return {}


class _FakeTrainer:
    def __init__(self):
        self.modules = _FakeModules()

    def set_callbacks(self, callbacks):
        pass


class _FakeEnvironment:
    observation_space = SimpleNamespace(shape=(4, 4, 3))
    single_action_space = object()

    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("env.close")


def _config(tmp_path):
    return SimpleNamespace(
        run=SimpleNamespace(
            log_level="INFO",
            logs_root=str(tmp_path),
            seed=7,
            total_steps=8,
            log_config=None,
        ),
        tracking=SimpleNamespace(
            verbosity="ALL",
            experiment_name="test",
            callbacks=[],
        ),
        environment=SimpleNamespace(id="test-env"),
        algorithm=SimpleNamespace(name="PPO"),
        model_dump=lambda **kwargs: {},
    )


@pytest.mark.parametrize(
    ("run_error", "expected_status"),
    [
        (None, "FINISHED"),
        (RuntimeError("training failed"), "FAILED"),
        (KeyboardInterrupt(), "KILLED"),
    ],
)
def test_run_training_closes_environment_before_metrics_logger(
    monkeypatch, tmp_path, run_error, expected_status
):
    events = []
    env = _FakeEnvironment(events)
    metrics_logger = _FakeMetricsLogger(events)
    trainer = _FakeTrainer()

    monkeypatch.setattr(train_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "resolve_session_id", lambda: "run-1")
    monkeypatch.setattr(train_module, "resolve_device", lambda config: "cpu")
    monkeypatch.setattr(train_module, "_seed_everything", lambda seed: None)
    monkeypatch.setattr(
        train_module, "make_env", lambda *args, **kwargs: env
    )
    monkeypatch.setattr(
        train_module, "BuildContext", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        train_module,
        "build_metrics_logger",
        lambda *args, **kwargs: metrics_logger,
    )
    monkeypatch.setattr(
        train_module, "build_ppo", lambda build_ctx: (object(), trainer)
    )
    monkeypatch.setattr(
        train_module, "build_callbacks", lambda *args, **kwargs: None
    )

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, total_steps):
            if run_error is not None:
                raise run_error

    monkeypatch.setattr(train_module, "RolloutCollector", _FakeCollector)

    if run_error is None:
        train_module.run_training(_config(tmp_path))
    else:
        with pytest.raises(type(run_error)):
            train_module.run_training(_config(tmp_path))

    assert events == ["env.close", ("metrics.close", expected_status)]


def test_composite_close_is_idempotent_and_rejects_late_writes():
    first_sink = Mock()
    second_sink = Mock()
    metrics_logger = CompositeMetricsLogger([first_sink, second_sink])

    metrics_logger.close(status="KILLED")
    metrics_logger.close(status="FINISHED")

    first_sink.close.assert_called_once_with(status="KILLED")
    second_sink.close.assert_called_once_with(status="KILLED")
    with pytest.raises(RuntimeError, match="closed"):
        metrics_logger.log_metrics({"loss": 1.0})


def test_composite_close_attempts_every_sink_after_a_failure():
    first_sink = Mock()
    second_sink = Mock()
    second_sink.close.side_effect = OSError("sink shutdown failed")
    metrics_logger = CompositeMetricsLogger([first_sink, second_sink])

    with pytest.raises(OSError, match="sink shutdown failed"):
        metrics_logger.close()

    first_sink.close.assert_called_once_with(status="FINISHED")


def test_composite_context_closes_with_interruption_status():
    sink = Mock()
    metrics_logger = CompositeMetricsLogger([sink])

    with pytest.raises(KeyboardInterrupt):
        with metrics_logger:
            raise KeyboardInterrupt()

    sink.close.assert_called_once_with(status="KILLED")


def test_metrics_logger_builder_closes_partial_sinks_on_failure(monkeypatch):
    mlflow_sink = Mock()
    monkeypatch.setattr(
        tracking_builder,
        "_build_mlflow_metrics_sink",
        lambda *args, **kwargs: mlflow_sink,
    )

    def fail_console_sink(*args, **kwargs):
        raise RuntimeError("console sink setup failed")

    monkeypatch.setattr(
        tracking_builder, "_build_console_metrics_sink", fail_console_sink
    )
    tracking_settings = SimpleNamespace(mlflow=True, console=True)

    with pytest.raises(RuntimeError, match="console sink setup failed"):
        tracking_builder.build_metrics_logger(tracking_settings, object())

    mlflow_sink.close.assert_called_once_with(status="FAILED")


def test_main_stops_logging_after_training_raises(monkeypatch):
    events = []

    def fail_training(*args, **kwargs):
        events.append("training")
        raise RuntimeError("failed")

    monkeypatch.setattr(train_module, "load_config", lambda path: object())
    monkeypatch.setattr(train_module, "run_training", fail_training)
    monkeypatch.setattr(
        train_module, "shutdown_logging", lambda: events.append("logging.stop")
    )

    with pytest.raises(RuntimeError, match="failed"):
        train_module.main("run.yaml")

    assert events == ["training", "logging.stop"]
