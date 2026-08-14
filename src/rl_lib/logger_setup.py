from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "logging.yaml"


def setup_logging(
    config_path: str | Path = _DEFAULT_CONFIG,
    default_level: int = logging.DEBUG,
) -> None:
    config_path = Path(config_path)

    if not config_path.is_file():
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).warning(
            "Logging config '%s' not found, falling back to basicConfig.", config_path
        )
        return

    with config_path.open("rt") as f:
        config = yaml.safe_load(f)

    for handler in config.get("handlers", {}).values():
        if "filename" in handler:
            Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
