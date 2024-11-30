"""Logger for the application."""

import os
import sys
from types import EllipsisType
from typing import Any, Callable

from loguru import logger as base_logger
from loguru._logger import Logger

from ai_api.util.singleton import Singleton


class MockLogger(metaclass=Singleton):
    """Mock logger for testing purposes."""

    def __getattr__(self, _: str) -> Callable[[Any], EllipsisType]:
        """Return a method that does nothing to mock logger methods."""
        return lambda *_, **__: ...


class SingletonLogger(metaclass=Singleton):
    """Singleton logger for the application."""

    def __init__(self) -> None:
        """Initialize the singleton logger."""
        self.logger = base_logger.bind(logger="main_logger")
        self.logger.add(sys.stdout,
                       level="INFO",
                       enqueue=True,
                       format="{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {level} | {message}")

def get_logger() -> Logger:
    """Get logger instance.

    :return: logger instance
    """
    if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST" in os.environ:  # pragma: no cover
        return MockLogger()

    return SingletonLogger().logger
