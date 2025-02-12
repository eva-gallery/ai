"""Module providing logging functionality for the application.

This module implements a thread-safe singleton logger using loguru, with special
handling for test environments. It provides consistent logging across the application
with configurable output formatting and log levels.
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import sys
import threading
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict

from loguru import logger as base_logger
from loguru._defaults import LOGURU_ERROR_NO
from loguru._recattrs import RecordFile, RecordLevel

from ai_api import settings
from ai_api.util.singleton import Singleton

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import EllipsisType

    from loguru import Record
    from loguru._logger import Logger
else:

    class Record(TypedDict):
        """Minimized version of loguru's Record."""

        file: RecordFile
        module: str
        level: RecordLevel
        line: int
        message: str
        time: datetime


# Compile regex pattern once at module level
_XML_PATTERN = re.compile(r"(<[^>\s]+>)")


class MockLogger(metaclass=Singleton):
    """Mock logger implementation for testing environments.

    This class provides a no-op logger that can be used in test environments
    to prevent actual logging operations. It implements the same interface
    as the real logger but does nothing.
    """

    def __getattr__(self, _: str) -> Callable[[Any], EllipsisType]:
        """Return a no-op method for any requested logger method.

        :param _: Name of the requested method (ignored).
        :type _: str
        :returns: No-op function that accepts any arguments.
        :rtype: Callable[[Any], EllipsisType]
        """
        return lambda *_, **__: ...


# Global lock for thread-safe initialization
_logger_lock = threading.Lock()
_logger_initialized = False


def _initialize_logging() -> None:
    """Initialize logging setup only once."""
    global _logger_initialized  # noqa: PLW0603

    with _logger_lock:
        if _logger_initialized:
            return

        class InterceptHandler(logging.Handler):
            """Intercepts standard library logging and redirects to loguru."""

            def emit(self, record: logging.LogRecord) -> None:
                """Process a log record by redirecting it to loguru."""
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                frame = logging.currentframe()
                depth = 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(
                    level,
                    record.getMessage(),
                )

        # Configure stdlib logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Configure warnings
        def showwarning(  # noqa: PLR0913
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            """Forward warnings to the logger."""
            warning_message = f"{category.__name__} at {filename}:{lineno}: {message}"
            logger.warning(warning_message)

            if settings.debug:  # type: ignore[attr-defined]
                _original_showwarning(message, category, filename, lineno, file, line)

        _original_showwarning = warnings.showwarning
        warnings.showwarning = showwarning

        _logger_initialized = True


class SingletonLogger(metaclass=Singleton):
    """Singleton logger for the application."""

    def __init__(self) -> None:
        """Initialize the logger with stdout output."""
        base_logger.remove()

        self.logger = base_logger.bind(logger="main_logger")
        self.error_logger = base_logger.bind(logger="error_logger")

        # Initialize logging exactly once
        _initialize_logging()

        # Add handler for regular logs (INFO and below)
        self.logger.add(
            sys.stdout,
            level=settings.log_level,
            enqueue=True,
            filter=lambda record: record["level"].no < LOGURU_ERROR_NO,  # type: ignore[attr-defined]
            format=self.preprocess_record,
            colorize=True,
        )

        # Add handler for error logs (ERROR and above)
        self.error_logger.add(
            sys.stdout,
            level="ERROR",
            enqueue=True,
            format=self.preprocess_record,
            colorize=True,
        )

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up logger resources on exit."""
        self.logger.remove()
        self.error_logger.remove()

    @staticmethod
    def preprocess_record(record: Record, *_: Any, **__: Any) -> str:
        """Preprocess message to remove newlines and escape special characters for logging.

        :param record: The log record to process.
        :param _: Additional positional arguments (ignored).
        :param __: Additional keyword arguments (ignored).
        :returns: Formatted log string.
        """
        message = str(record.get("message", ""))

        if message:
            # Replace newlines with visible markers
            message = message.replace("\r\n", "␤").replace("\n\r", "␤").replace("\n", "␤").replace("\r", "␤")
            # Escape quotes to prevent format string issues
            message = message.replace('"', '\\"').replace("'", "\\'")
            # Escape XML-like patterns using pre-compiled pattern
            message = _XML_PATTERN.sub(r"\\\1", message)

        return "<green>{time}</green> | <cyan>{file}:{line}</cyan> | <level>{level}</level> | <level>{message}</level>\n".format(
            time=record.get("time", datetime.now(tz=UTC)).strftime("%Y-%m-%d %H:%M:%S"),
            file=record.get("file", RecordFile(name="", path="")).name,
            line=record.get("line", 0),
            level=record.get("level", RecordLevel(name="", no=0, icon="")).name,
            message=message,
        )


# Global logger instance
logger = SingletonLogger().logger if "PYTEST_CURRENT_TEST" not in os.environ else MockLogger()


def get_logger() -> Logger:
    """Get the appropriate logger instance based on environment.

    :returns: Logger instance appropriate for the current environment.
    """
    return logger


if __name__ == "__main__":
    logger.debug("Logger pre-flight check with standalone <tag> and also <tag>with closing</tag>")
