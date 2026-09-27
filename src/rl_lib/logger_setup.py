"""Logging configuration with session ID injection and queue-based async handlers.

Configures Python logging from a YAML file (default: config/logging.yaml),
injects a unique session ID into every log record, and ensures proper
shutdown of queue listeners on exit.
"""

from __future__ import annotations

import atexit
import logging
import logging.config
import logging.handlers
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "logging.yaml"


def _generate_session_id() -> str:
    """Generate a unique session identifier.

    Format: <hostname>-<UTC timestamp>-<short uuid>, matching MLflowLogger's
    run naming scheme for easy correlation between log files and MLflow runs.

    Returns:
        Session ID string.
    """
    host = socket.gethostname().split(".")[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:6]
    return f"{host}-{ts}-{short_id}"


def _install_session_id_factory(session_id: str) -> None:
    """Wrap the global LogRecord factory to inject session_id into every record.

    Args:
        session_id: The session ID to attach to all log records.
    """
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.session_id = session_id
        return record

    logging.setLogRecordFactory(record_factory)


def setup_logging(
    config_path: str | Path = _DEFAULT_CONFIG,
    default_level: int = logging.DEBUG,
    session_id: str | None = None,
) -> str:
    """Configure logging from YAML, inject session ID, start queue listener.

    Args:
        config_path: Path to logging YAML config. Defaults to package config.
        default_level: Fallback level if config file not found.
        session_id: Explicit session ID (overrides env var and generation).

    Returns:
        The session ID used (generated or provided).

    Side Effects:
        - Modifies global logging configuration
        - Starts queue handler listener thread
        - Registers shutdown_logging at exit
    """
    session_id = session_id or os.environ.get("RL_LIB_SESSION_ID") or _generate_session_id()
    _install_session_id_factory(session_id)

    config_path = Path(config_path)

    if not config_path.is_file():
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).warning(
            "Logging config '%s' not found, falling back to basicConfig.", config_path
        )
        return session_id

    with config_path.open("rt") as f:
        config = yaml.safe_load(f)

    for handler in config.get("handlers", {}).values():
        if "filename" in handler:
            Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)

    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None and queue_handler.listener is not None:
        queue_handler.listener.start()

    atexit.register(shutdown_logging)
    return session_id


def shutdown_logging() -> None:
    """Stop the queue handler listener thread on process exit.

    Safe to call multiple times; checks listener state before stopping.
    """
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None and queue_handler.listener is not None and queue_handler.listener._thread is not None:
        queue_handler.listener.stop()
