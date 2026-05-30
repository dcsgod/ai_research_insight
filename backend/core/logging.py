"""
Structured logging configuration for the AI Research Intelligence Platform.

Provides:
- JSON-structured logs for production (machine-readable)
- Human-friendly colourised console logs for development
- Request correlation IDs propagated through async context
- Configurable log levels and output destinations

Usage::

    from backend.core.logging import setup_logging, get_logger

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Server started", port=8000)
"""

from __future__ import annotations

import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Correlation ID context variable
# ---------------------------------------------------------------------------
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the current request's correlation ID, or empty string if none."""
    return _correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current async context.

    Args:
        cid: Explicit correlation ID; generates a new UUID4 if omitted.

    Returns:
        The correlation ID that was set.
    """
    cid = cid or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Reset the correlation ID for the current async context."""
    _correlation_id.set("")


# ---------------------------------------------------------------------------
# Custom JSON formatter (stdlib-only, no structlog required)
# ---------------------------------------------------------------------------
import json
import traceback
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Each record includes:
    - timestamp (ISO-8601, UTC)
    - level
    - logger name
    - message
    - correlation_id (from context var)
    - extra key-value pairs
    - exception info if present
    """

    RESERVED_ATTRS = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()

        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "correlation_id": get_correlation_id() or None,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Merge extra fields provided by the caller
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_dict[key] = value

        # Append exception traceback if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_dict["exception"] = record.exc_text

        if record.stack_info:
            log_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_dict, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable colourised formatter for development.

    Colours:
    - DEBUG   → cyan
    - INFO    → green
    - WARNING → yellow
    - ERROR   → red
    - CRITICAL → bright red
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    COLOURS: dict[str, str] = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
        "CRITICAL": "\033[1;31m",  # bold red
    }

    FMT = (
        "{colour}{bold}[{levelname:<8}]{reset} "
        "\033[90m{asctime}\033[0m  "
        "\033[35m{name}\033[0m  "
        "{message}"
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self.COLOURS.get(record.levelname, self.RESET)
        formatted_time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        cid = get_correlation_id()
        prefix = f"[{cid[:8]}] " if cid else ""

        line = self.FMT.format(
            colour=colour,
            bold=self.BOLD,
            reset=self.RESET,
            levelname=record.levelname,
            asctime=formatted_time,
            name=record.name,
            message=f"{prefix}{record.getMessage()}",
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _build_logging_config(
    log_level: str,
    log_format: str,
    log_file: Optional[str],
    is_development: bool,
) -> dict[str, Any]:
    """Build the logging.config.dictConfig dictionary."""
    use_json = log_format == "json" or not is_development

    formatters: dict[str, Any] = {
        "json": {
            "()": JSONFormatter,
        },
        "console": {
            "()": ConsoleFormatter,
        },
    }

    chosen_formatter = "json" if use_json else "console"

    handlers: dict[str, Any] = {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": chosen_formatter,
            "level": log_level,
        },
    }

    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 100 * 1024 * 1024,  # 100 MB
            "backupCount": 5,
            "formatter": "json",  # always JSON to file
            "level": log_level,
            "encoding": "utf-8",
        }

    root_handlers = list(handlers.keys())

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            # Application loggers
            "backend": {
                "level": log_level,
                "handlers": root_handlers,
                "propagate": False,
            },
            # Third-party loggers — reduce noise
            "uvicorn": {
                "level": "INFO",
                "handlers": root_handlers,
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "WARNING",
                "handlers": root_handlers,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "ERROR",
                "handlers": root_handlers,
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": root_handlers,
                "propagate": False,
            },
            "sqlalchemy.pool": {
                "level": "WARNING",
                "handlers": root_handlers,
                "propagate": False,
            },
            "httpx": {
                "level": "WARNING",
                "handlers": root_handlers,
                "propagate": False,
            },
            "aioredis": {
                "level": "WARNING",
                "handlers": root_handlers,
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": root_handlers,
        },
    }


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    is_development: Optional[bool] = None,
) -> None:
    """
    Initialise application logging.

    Falls back to settings from ``get_settings()`` when arguments are not
    explicitly provided, making this safe to call before or after settings
    are fully configured.

    Args:
        log_level:      Override ``LOG_LEVEL`` setting (e.g. "DEBUG").
        log_format:     Override ``LOG_FORMAT`` setting ("json" | "console").
        log_file:       Override ``LOG_FILE`` setting (path string or None).
        is_development: Override environment detection.

    Example::

        # In main.py lifespan or module level:
        setup_logging()
    """
    try:
        from backend.core.config import get_settings

        settings = get_settings()
        _log_level = log_level or settings.LOG_LEVEL
        _log_format = log_format or settings.LOG_FORMAT
        _log_file = log_file or settings.LOG_FILE
        _is_development = is_development if is_development is not None else settings.is_development
    except Exception:
        # Fallback for early bootstrap (before .env is loaded)
        _log_level = log_level or "INFO"
        _log_format = log_format or "console"
        _log_file = log_file
        _is_development = is_development if is_development is not None else True

    config = _build_logging_config(
        log_level=_log_level,
        log_format=_log_format,
        log_file=_log_file,
        is_development=_is_development,
    )
    logging.config.dictConfig(config)

    root_logger = logging.getLogger("backend")
    root_logger.debug(
        "Logging initialised",
        extra={
            "log_level": _log_level,
            "log_format": _log_format,
            "log_file": _log_file,
            "is_development": _is_development,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper to retrieve a namespaced logger.

    The ``name`` argument should be ``__name__`` of the calling module so
    that log hierarchy and filtering work correctly.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.

    Example::

        logger = get_logger(__name__)
        logger.info("Processing paper", extra={"arxiv_id": "2301.00001"})
    """
    return logging.getLogger(name)
