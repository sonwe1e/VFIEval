from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig


LOGGER_NAME = "vfieval.runtime"
_CONFIG_LOCK = threading.Lock()
_CONFIGURED_LOG_PATH: Path | None = None
_LOGGER = logging.getLogger(LOGGER_NAME)
_LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False


def _redact_text(value: str, roots: tuple[str, ...]) -> str:
    result = value
    for root in roots:
        if root:
            result = result.replace(root, "<workspace>")
            result = result.replace(root.replace("\\", "/"), "<workspace>")
    result = re.sub(r"(/evaluate/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/api/blind/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/tasks/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/reviews/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"([?&](?:token|key|secret|signature)=)[^&\s]+", r"\1<redacted>", result, flags=re.I)
    result = re.sub(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", result, flags=re.I)
    result = re.sub(
        r"(\b(?:token|secret|password|authorization|cookie|api_key|access_key|private_key)\s*[:=]\s*)"
        r"([^\s,;}]+)",
        r"\1<redacted>",
        result,
        flags=re.I,
    )
    return result


def _safe_value(value: Any, roots: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, roots)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if any(
                part in name.lower()
                for part in (
                    "token",
                    "secret",
                    "password",
                    "authorization",
                    "cookie",
                    "api_key",
                    "access_key",
                    "private_key",
                )
            ):
                sanitized[name] = "<redacted>"
            else:
                sanitized[name] = _safe_value(child, roots)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_safe_value(child, roots) for child in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value), roots)


class _JsonFormatter(logging.Formatter):
    def __init__(self, roots: tuple[str, ...]):
        super().__init__()
        self.roots = roots

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": str(getattr(record, "event", record.name)),
            "message": record.getMessage(),
        }
        details = getattr(record, "details", None)
        if isinstance(details, dict):
            payload.update(details)
        if record.exc_info:
            payload["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(_safe_value(payload, self.roots), ensure_ascii=False, sort_keys=True)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, roots: tuple[str, ...]):
        super().__init__()
        self.roots = roots

    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", record.name))
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = _redact_text(record.getMessage(), self.roots)
        return f"[{stamp}] {record.levelname:<7} {event}: {message}"


def configure_runtime_logging(
    workspace: WorkspaceConfig,
    *,
    filename: str = "server.jsonl",
) -> Path:
    """Configure a process-wide rotating JSONL diagnostic log.

    Reconfiguration is intentional for embedded servers and tests that use a
    different workspace in the same Python process.
    """

    global _CONFIGURED_LOG_PATH
    log_dir = workspace.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(str(filename or "server.jsonl")).name
    if not safe_filename.endswith(".jsonl"):
        raise ValueError("runtime log filename must end with .jsonl")
    log_path = (log_dir / safe_filename).resolve()
    with _CONFIG_LOCK:
        logger = logging.getLogger(LOGGER_NAME)
        if _CONFIGURED_LOG_PATH == log_path and logger.handlers:
            return log_path
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        roots = tuple(
            sorted(
                {str(workspace.root.resolve()), str(workspace.root.parent.resolve())},
                key=len,
                reverse=True,
            )
        )
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonFormatter(roots))
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(_ConsoleFormatter(roots))
        logger.handlers[:] = [file_handler, console_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED_LOG_PATH = log_path
    return log_path


def runtime_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(level: int, event: str, message: str, **details: Any) -> None:
    runtime_logger().log(level, message, extra={"event": event, "details": details})


def close_runtime_logging() -> None:
    global _CONFIGURED_LOG_PATH
    with _CONFIG_LOCK:
        logger = logging.getLogger(LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.flush()
            finally:
                handler.close()
        _CONFIGURED_LOG_PATH = None
