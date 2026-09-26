"""Unit tests for the small utility modules: the buffer/tensor conversion, the
logging bootstrap and the MLflow backend.

MLflow is stubbed out — the real client took ~a minute to start a run, and the
contract under test is which MLflow call is issued with which arguments.
"""

from __future__ import annotations

import importlib
import logging
import logging.config
import logging.handlers
import re
import socket

import numpy as np
import pytest
import torch as T

from rl_lib.buffers.utils import to_tensor_batch
from rl_lib.logger_setup import (
    _generate_session_id,
    setup_logging,
    shutdown_logging,
)
from rl_lib.tracking.mlflow_logger import MLflowLogger


# ---------------------------------------------------------------- to_tensor_batch


def test_to_tensor_batch_converts_every_value():
    batch = {
        "observation": np.zeros((2, 4, 4, 1), dtype=np.uint8),
        "action": np.ones((2, 3), dtype=np.float32),
        "dones": np.array([True, False]),
    }

    converted = to_tensor_batch(batch, T.device("cpu"))

    assert set(converted) == set(batch)
    assert all(isinstance(value, T.Tensor) for value in converted.values())
    assert converted["observation"].dtype == T.uint8
    assert converted["action"].dtype == T.float32
    assert converted["dones"].dtype == T.bool
    T.testing.assert_close(converted["action"], T.ones(2, 3))


def test_to_tensor_batch_shares_memory_with_the_source_arrays():
    """`T.from_numpy` shares the buffer, so the conversion is free but mutating
    the tensor is visible in the numpy array."""

    batch = {"action": np.ones((2, 3), dtype=np.float32)}

    converted = to_tensor_batch(batch, T.device("cpu"))
    converted["action"].fill_(0.0)

    assert batch["action"].sum() == 0.0


def test_to_tensor_batch_returns_an_empty_dict_for_an_empty_batch():
    assert to_tensor_batch({}, T.device("cpu")) == {}


def test_to_tensor_batch_accepts_a_device_index():
    converted = to_tensor_batch({"a": np.zeros(1, np.float32)}, "cpu")

    assert converted["a"].device.type == "cpu"


# ---------------------------------------------------------------- logging setup


@pytest.fixture
def restore_logging():
    """`setup_logging` reconfigures the root logger; put it back afterwards."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    factories = [logging.getLogRecordFactory()]
    try:
        yield
    finally:
        logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})
        root = logging.getLogger()
        for handler in list(root.handlers):
            if hasattr(handler, "listener") and handler.listener is not None:
                handler.listener.stop()
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        logging.setLogRecordFactory(factories[0])


@pytest.fixture
def logging_config(tmp_path) -> str:
    config = tmp_path / "logging.yaml"
    config.write_text(
        f"""
version: 1
disable_existing_loggers: false
formatters:
  standard:
    format: "%(session_id)s | %(name)s | %(message)s"
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: {tmp_path / "nested" / "app.log"}
    maxBytes: 1024
    backupCount: 1
  queue_handler:
    class: logging.handlers.QueueHandler
    handlers: [file]
root:
  level: DEBUG
  handlers: [queue_handler]
""".lstrip()
    )
    return str(config)


def test_generate_session_id_uses_the_documented_scheme():
    session_id = _generate_session_id()

    host, _, short_id = session_id.rpartition("-")
    timestamp = host[host.rindex("-") + 1 :]
    assert host[: host.rindex("-")] == socket.gethostname().split(".")[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", timestamp)
    assert re.fullmatch(r"[0-9a-f]{6}", short_id)


def test_setup_logging_returns_the_explicit_session_id(restore_logging, logging_config):
    assert setup_logging(logging_config, session_id="run-42") == "run-42"


def test_setup_logging_reads_the_session_id_from_the_environment(
    restore_logging, logging_config, monkeypatch
):
    monkeypatch.setenv("RL_LIB_SESSION_ID", "from-env")

    assert setup_logging(logging_config) == "from-env"


def test_setup_logging_stamps_every_record_with_the_session_id(
    restore_logging, logging_config
):
    setup_logging(logging_config, session_id="run-42")
    logger = logging.getLogger("rl_lib.test")

    logger.info("hello")

    shutdown_logging()
    records = [
        record
        for handler in logging.getLogger().handlers
        for record in getattr(handler, "records", [])
    ]
    assert logging.getLogRecordFactory()(  # the factory is the installed one
        "n", logging.INFO, "p", 1, "m", None, None
    ).session_id == "run-42"
    assert records == [] or all(record.session_id == "run-42" for record in records)


def test_setup_logging_creates_the_log_directory(restore_logging, logging_config, tmp_path):
    setup_logging(logging_config, session_id="run-42")

    assert (tmp_path / "nested").is_dir()
    shutdown_logging()


def test_setup_logging_starts_the_queue_listener(restore_logging, logging_config):
    setup_logging(logging_config, session_id="run-42")

    queue_handler = logging.getHandlerByName("queue_handler")
    assert queue_handler is not None
    assert queue_handler.listener is not None
    assert queue_handler.listener._thread is not None

    shutdown_logging()
    assert queue_handler.listener._thread is None


def test_setup_logging_falls_back_when_the_config_is_missing(
    restore_logging, tmp_path, caplog
):
    with caplog.at_level(logging.WARNING, logger="rl_lib.logger_setup"):
        session_id = setup_logging(tmp_path / "absent.yaml")

    assert re.fullmatch(r".+-\d{8}T\d{6}Z-[0-9a-f]{6}", session_id)
    assert "not found, falling back to basicConfig" in caplog.text
    assert logging.getHandlerByName("queue_handler") is None


def test_shutdown_logging_without_a_queue_handler_is_a_noop(restore_logging):
    logging.getLogger().handlers.clear()

    shutdown_logging()


# ---------------------------------------------------------------- mlflow backend


class _FakeMlflowPytorch:
    def __init__(self, calls: list):
        self._calls = calls

    def log_model(self, *args, **kwargs):
        self._calls.append(("log_model", args, kwargs))

    def log_state_dict(self, *args, **kwargs):
        self._calls.append(("log_state_dict", args, kwargs))


class _FakeMlflow:
    def __init__(self):
        self.calls: list = []
        self.pytorch = _FakeMlflowPytorch(self.calls)

    def set_experiment(self, name):
        self.calls.append(("set_experiment", (name,), {}))

    def start_run(self, **kwargs):
        self.calls.append(("start_run", (), kwargs))
        return object()

    def log_metrics(self, metrics, step=None):
        self.calls.append(("log_metrics", (metrics,), {"step": step}))

    def log_params(self, params):
        self.calls.append(("log_params", (params,), {}))

    def log_artifact(self, path, artifact_path=None):
        self.calls.append(("log_artifact", (path,), {"artifact_path": artifact_path}))

    def end_run(self, status=None):
        self.calls.append(("end_run", (), {"status": status}))

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture
def fake_mlflow(monkeypatch):
    module = importlib.import_module("rl_lib.tracking.mlflow_logger")
    fake = _FakeMlflow()
    monkeypatch.setattr(module, "mlflow", fake)
    return fake


@pytest.fixture
def logger(fake_mlflow) -> MLflowLogger:
    return MLflowLogger(experiment_name="test-experiment", run_name="run-1")


def test_constructor_starts_the_experiment_and_the_run(fake_mlflow):
    MLflowLogger(experiment_name="test-experiment", run_name="run-1")

    assert fake_mlflow.names() == ["set_experiment", "start_run"]
    assert fake_mlflow.calls[0][1] == ("test-experiment",)
    assert fake_mlflow.calls[1][2] == {"run_name": "run-1", "log_system_metrics": True}


def test_constructor_deduces_a_run_name_when_none_is_given(fake_mlflow):
    MLflowLogger(experiment_name="test-experiment")

    run_name = fake_mlflow.calls[1][2]["run_name"]
    assert re.fullmatch(r".+-\d{8}T\d{6}Z-[0-9a-f]{6}", run_name)


def test_log_metrics_forwards_the_step(logger: MLflowLogger, fake_mlflow):
    logger.log_metrics({"loss/total": 1.5}, step=7)

    assert fake_mlflow.calls[-1] == ("log_metrics", ({"loss/total": 1.5},), {"step": 7})


def test_log_parameters_stringifies_the_values(logger: MLflowLogger, fake_mlflow):
    logger.log_parameters({"lr": 1e-4, "optimizer": "AdamW", "flag": True})

    assert fake_mlflow.calls[-1] == (
        "log_params",
        ({"lr": "0.0001", "optimizer": "AdamW", "flag": "True"},),
        {},
    )


def test_log_model_uses_the_registered_model_name(logger: MLflowLogger, fake_mlflow):
    logger._registered_model_name = "policy-v1"
    model = T.nn.Linear(2, 2)

    logger.log_model(model, artifact_path="net")

    name, args, kwargs = fake_mlflow.calls[-1]
    assert name == "log_model"
    assert args == (model,)
    assert kwargs == {"artifact_path": "net", "registered_model_name": "policy-v1"}


def test_log_model_registered_name_can_be_overridden(logger: MLflowLogger, fake_mlflow):
    logger._registered_model_name = "policy-v1"

    logger.log_model(T.nn.Linear(2, 2), registered_model_name="other")

    assert fake_mlflow.calls[-1][2]["registered_model_name"] == "other"


def test_log_state_dict(logger: MLflowLogger, fake_mlflow):
    state_dict = {"weight": T.ones(2)}

    logger.log_state_dict(state_dict)

    name, args, kwargs = fake_mlflow.calls[-1]
    assert name == "log_state_dict"
    assert args == (state_dict,)
    assert kwargs == {"artifact_path": "checkpoints"}


def test_log_artifact(logger: MLflowLogger, fake_mlflow):
    logger.log_artifact("/tmp/video.mp4", artifact_path="videos/step_0")

    assert fake_mlflow.calls[-1] == (
        "log_artifact",
        ("/tmp/video.mp4",),
        {"artifact_path": "videos/step_0"},
    )


def test_close_ends_the_run_with_the_given_status(logger: MLflowLogger, fake_mlflow):
    logger.close(status="FAILED")

    assert fake_mlflow.calls[-1] == ("end_run", (), {"status": "FAILED"})


def test_close_defaults_to_finished(logger: MLflowLogger, fake_mlflow):
    logger.close()

    assert fake_mlflow.calls[-1] == ("end_run", (), {"status": "FINISHED"})


def test_context_manager_closes_the_run_on_success(logger: MLflowLogger, fake_mlflow):
    with logger as entered:
        assert entered is logger

    assert fake_mlflow.calls[-1] == ("end_run", (), {"status": "FINISHED"})


def test_context_manager_closes_the_run_as_failed_on_error(
    logger: MLflowLogger, fake_mlflow
):
    with pytest.raises(RuntimeError):
        with logger:
            raise RuntimeError("boom")

    assert fake_mlflow.calls[-1] == ("end_run", (), {"status": "FAILED"})


def test_logger_satisfies_the_metrics_logger_interface(logger: MLflowLogger):
    from rl_lib.tracking.base import MetricsLogger

    assert isinstance(logger, MetricsLogger)
