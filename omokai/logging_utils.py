from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any


LOGGER_ROOT_NAME = "omokai"
_BASE_RECORD = logging.makeLogRecord({})
_STANDARD_RECORD_FIELDS = set(_BASE_RECORD.__dict__.keys()) | {"message", "asctime"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


class JsonlFileHandler(logging.Handler):
    def __init__(self, path: str | Path, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        event = getattr(record, "omokai_event", None)
        if event is not None:
            payload["event"] = _json_safe(event)
        fields = getattr(record, "omokai_fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[str(key)] = _json_safe(value)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
            and key not in {"omokai_event", "omokai_fields", "exc_info", "exc_text", "stack_info"}
        }
        if extras:
            payload["extra"] = _json_safe(extras)
        if record.exc_info:
            payload["exception"] = logging.Formatter().formatException(record.exc_info)
        line = json.dumps(payload, ensure_ascii=False)
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            self.handleError(record)


def configure_debug_logging(
    path: str | Path,
    jsonl_path: str | Path | None = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    structured_log_path = Path(jsonl_path) if jsonl_path is not None else log_path.with_suffix(".jsonl")
    structured_log_path.parent.mkdir(parents=True, exist_ok=True)

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

    jsonl_handler = JsonlFileHandler(structured_log_path, level=level)
    jsonl_handler._omokai_managed = True  # type: ignore[attr-defined]
    logger.addHandler(jsonl_handler)
    return logger


def get_debug_logger(name: str | None = None) -> logging.Logger:
    logger_name = LOGGER_ROOT_NAME if not name else f"{LOGGER_ROOT_NAME}.{name}"
    return logging.getLogger(logger_name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *args: object,
    **fields: object,
) -> None:
    logger.log(
        level,
        message,
        *args,
        extra={
            "omokai_event": event,
            "omokai_fields": fields,
        },
    )
