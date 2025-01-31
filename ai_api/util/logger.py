"""Module providing logging functionality for the application.

This module implements a thread-safe singleton logger using loguru, with special
handling for test environments. It provides consistent logging across the application
with configurable output formatting and log levels.
"""

import os
import sys
from collections.abc import Callable
from types import EllipsisType
from typing import Any

from loguru import logger as base_logger
from loguru._logger import Logger

from ai_api import settings
from ai_api.util.singleton import Singleton


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


class SingletonLogger(metaclass=Singleton):
    """Thread-safe singleton logger implementation.

    This class provides a thread-safe singleton logger using loguru. It configures
    the logger with standard output and formatting options. The singleton pattern
    ensures consistent logging behavior across the application.
    """

    def __init__(self) -> None:
        """Initialize the singleton logger with standard configuration.

        Configures the logger to output to stdout with timestamp, file location,
        log level, and message formatting.
        """
        self.logger = base_logger.bind(logger="main_logger")
        self.logger.add(sys.stdout,
                       level=settings.log_level,
                       enqueue=True,
                       format="{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {level} | {message}")


def get_logger() -> Logger:
    """Get the appropriate logger instance based on environment.

    Returns a mock logger in test environments and the real singleton logger
    in all other environments.

    :returns: Logger instance appropriate for the current environment.
    :rtype: Logger
    """
    if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST" in os.environ or os.environ.get("CI"):  # pragma: no cover
        return MockLogger()

    return SingletonLogger().logger
