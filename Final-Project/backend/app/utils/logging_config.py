"""Logging setup used by the application."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format log messages as compact JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
        )


def configure_logging(level: str = "INFO") -> None:
    """Initialize application-wide logging."""

    root_logger = logging.getLogger()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
