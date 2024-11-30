"""Typeguards for the application."""

from typing import TypeGuard

from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session as SyncSessionType
from sqlalchemy.orm.session import SessionTransaction


def is_async_session(value: object) -> TypeGuard["AsyncSessionType"]:
    """Check if a value is an async session."""
    return isinstance(value, AsyncSessionType)


def is_sync_session(value: object) -> TypeGuard["SyncSessionType"]:
    """Check if a value is a sync session."""
    return isinstance(value, SyncSessionType)


def is_sync_session_transaction(value: object) -> TypeGuard["SessionTransaction"]:
    """Check if a value is a sync session transaction."""
    return isinstance(value, SessionTransaction)


def is_str(value: object) -> TypeGuard[str]:
    """Check if a value is a string."""
    return isinstance(value, str)


def is_list_of_str(value: object) -> TypeGuard[list[str]]:
    """Check if a value is a list of strings."""
    return isinstance(value, list) and all(isinstance(s, str) for s in value)
