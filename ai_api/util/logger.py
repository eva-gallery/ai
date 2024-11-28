import os
import sys
from typing import Any

from loguru import logger as base_logger

from ai_api.util.singleton import Singleton


class MockLogger(metaclass=Singleton):
    """Mock logger for testing purposes."""

    def __getattr__(self, item: str):
        def method(*args: Any, **kwargs: Any):
            pass
        return method


class SingletonLogger(metaclass=Singleton):
    """Singleton logger for the application."""

    def __init__(self):
        self.logger = base_logger.bind(logger="main_logger")
        self.logger.add(sys.stdout,
                       level="INFO",
                       enqueue=True,
                       format="{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {level} | {message}")

def get_logger():
    """Get logger instance.

    :return: logger instance
    """
    if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST" in os.environ:  # pragma: no cover
        return MockLogger()

    return SingletonLogger().logger
