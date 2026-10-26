"""Structured logging facility for SS_PROJECT_VCR_FRESH.

Outputs formatted logs with timestamps, module tags, severity, and optional fault codes.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from app.constants import FaultCode


class RobotLogFormatter(logging.Formatter):
    """Clean, structured log formatter for robotics operations."""

    def format(self, record: logging.LogRecord) -> str:
        fault_code = getattr(record, "fault_code", None)
        fault_tag = f" [{fault_code}]" if fault_code and fault_code != FaultCode.OK else ""
        record.fault_tag = fault_tag
        return super().format(record)


_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s]%(fault_tag)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize structured logging once at application entry."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = RobotLogFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)

    root = logging.getLogger("vcr")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(module_name: str) -> logging.Logger:
    """Obtain a structured logger under the 'vcr' namespace."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"vcr.{module_name}")


def log_fault(logger: logging.Logger, fault_code: FaultCode, message: str, level: int = logging.ERROR) -> None:
    """Helper to emit a structured log record tagged with a specific FaultCode."""
    logger.log(level, message, extra={"fault_code": fault_code.value})
