from __future__ import annotations

import logging
from pathlib import Path


LOGGER_ROOT_NAME = "omokai"


def configure_debug_logging(path: str | Path, level: int = logging.DEBUG) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_omokai_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler._omokai_managed = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def get_debug_logger(name: str | None = None) -> logging.Logger:
    logger_name = LOGGER_ROOT_NAME if not name else f"{LOGGER_ROOT_NAME}.{name}"
    return logging.getLogger(logger_name)
