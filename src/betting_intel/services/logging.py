"""
Structured logging with Loguru — production-grade logging for the system.
Provides JSON logging for Docker/production and pretty logging for development.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _base_logger

from betting_intel.config import settings


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level override (default: from settings)
        log_format: 'json' for structured JSON, 'pretty' for human-readable
        log_file: Optional path to log file
    """
    # Remove default handler
    _base_logger.remove()

    log_level = (level or settings.log_level).upper()
    fmt = log_format or settings.log_format

    # Console handler
    if fmt == "json":
        console_format = (
            '{{"timestamp":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level.name}",'
            '"module":"{module}",'
            '"function":"{function}",'
            '"line":{line},'
            '"message":"{message}",'
            '"extra":{extra}}}'
        )
        _base_logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            serialize=False,
            enqueue=True,
        )
    else:
        _base_logger.add(
            sys.stdout,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
            enqueue=True,
        )

    # File handler
    log_file_path = log_file or settings.log_file
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        _base_logger.add(
            str(log_path),
            format=(
                '{{"timestamp":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
                '"level":"{level.name}",'
                '"module":"{module}",'
                '"message":"{message}"}}'
            ),
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enqueue=True,
        )

    _base_logger.info(
        "Logging configured",
        level=log_level,
        format=fmt,
        log_file=str(log_file_path) if log_file_path else None,
    )


# Global logger instance
logger = _base_logger


def get_logger(name: str | None = None):
    """Get a logger instance. Backward-compatible alias for loguru.logger.bind."""
    if name:
        return _base_logger.bind(name=name)
    return _base_logger
