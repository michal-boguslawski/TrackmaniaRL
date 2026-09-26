"""Unit coverage for how the MLflow run status reflects the exception that ended a run.

The entrypoint must not swallow `BaseException`, so a failed or interrupted
training run has to reach MLflow as `FAILED`/`KILLED` and still propagate to
the process exit status. These tests pin both halves of that contract.
"""

import pytest

from rl_lib.tracking.mlflow_logger import MLflowLogger, run_status_for_exception


@pytest.mark.parametrize(
    ("exc_type", "exc_val", "expected"),
    [
        (None, None, "FINISHED"),
        (ValueError, ValueError("boom"), "FAILED"),
        (RuntimeError, RuntimeError("boom"), "FAILED"),
        (MemoryError, MemoryError(), "FAILED"),
        (KeyboardInterrupt, KeyboardInterrupt(), "KILLED"),
        (SystemExit, SystemExit(0), "FINISHED"),
        (SystemExit, SystemExit(2), "FAILED"),
    ],
)
def test_run_status_for_exception(exc_type, exc_val, expected):
    assert run_status_for_exception(exc_type, exc_val) == expected


def _logger_without_run():
    """An MLflowLogger with no live run, so `close` is the only thing under test."""
    logger = object.__new__(MLflowLogger)
    logger._registered_model_name = None
    return logger


@pytest.mark.parametrize(
    ("exc_type", "exc_val", "expected"),
    [
        (None, None, "FINISHED"),
        (RuntimeError, RuntimeError("boom"), "FAILED"),
        (KeyboardInterrupt, KeyboardInterrupt(), "KILLED"),
    ],
)
def test_exit_closes_run_with_matching_status(monkeypatch, exc_type, exc_val, expected):
    recorded = []
    monkeypatch.setattr(MLflowLogger, "close", lambda self, status: recorded.append(status))
    logger = _logger_without_run()

    logger.__exit__(exc_type, exc_val, None)

    assert recorded == [expected]


def test_exit_does_not_suppress_exception(monkeypatch):
    """`__exit__` must return falsy, or the entrypoint's re-raise is pointless."""
    monkeypatch.setattr(MLflowLogger, "close", lambda self, status: None)
    logger = _logger_without_run()

    assert not logger.__exit__(RuntimeError, RuntimeError("boom"), None)


def test_exit_propagates_exception_through_with_block(monkeypatch):
    monkeypatch.setattr(MLflowLogger, "close", lambda self, status: None)
    logger = _logger_without_run()

    with pytest.raises(RuntimeError, match="boom"):
        with logger:
            raise RuntimeError("boom")
