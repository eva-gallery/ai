"""Module containing utility functions and classes.

This module provides various utility functions and classes used throughout the API:
- Type checking and validation guards
- Logging configuration and utilities
- Singleton pattern implementation
- Session management utilities

These utilities help maintain code quality, type safety, and proper resource management.
"""

from ai_api.util.guard import (
    is_async_session,
    is_list_of_str,
    is_str,
    is_sync_session,
    is_sync_session_transaction,
)
from ai_api.util.logger import get_logger
from ai_api.util.singleton import Singleton

__all__ = ["Singleton", "get_logger", "is_async_session", "is_list_of_str", "is_str", "is_sync_session", "is_sync_session_transaction"]
