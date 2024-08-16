import sys

from loguru import logger as base_logger


def get_logger():  # TODO make into OTL-compliant logger; like Loguru
    logger = base_logger.bind(logger="main_logger")
    target = sys.stdout
    logger.add(target, level="DEBUG", enqueue=True, format="{file}:{line} | {level} | {message}")

    return logger
