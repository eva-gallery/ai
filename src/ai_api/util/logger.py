import sys

from loguru import logger as base_logger


def get_logger():
    logger = base_logger.bind(logger="main_logger")
    target = sys.stdout
    logger.add(target, level="debug", rotation="10MB", enqueue=True, format="{file}:{line} | {level} | {message}")

    return logger
