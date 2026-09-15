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
    """<hostname>-<UTC timestamp>-<short uuid>, same scheme as MLflowLogger's
    run name so you can correlate a log file with its MLflow run at a glance."""
    host = socket.gethostname().split(".")[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:6]
    return f"{host}-{ts}-{short_id}"


def _install_session_id_factory(session_id: str) -> None:
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
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None and queue_handler.listener is not None and queue_handler.listener._thread is not None:
        queue_handler.listener.stop()
