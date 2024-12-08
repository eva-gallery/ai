"""Module containing type guard functions for runtime type checking.

This module provides type guard functions that help ensure type safety at runtime
by checking if values match their expected types. These guards are particularly
useful for SQLAlchemy session handling and string type validation.
"""

from typing import TypeGuard

from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session as SyncSessionType
from sqlalchemy.orm.session import SessionTransaction


def is_async_session(value: object) -> TypeGuard["AsyncSessionType"]:
    """Check if a value is an SQLAlchemy async session.

    :param value: Value to check.
    :type value: object
    :returns: True if value is an async session.
    :rtype: TypeGuard[AsyncSessionType]
    """
    return isinstance(value, AsyncSessionType)


def is_sync_session(value: object) -> TypeGuard["SyncSessionType"]:
    """Check if a value is an SQLAlchemy sync session.

    :param value: Value to check.
    :type value: object
    :returns: True if value is a sync session.
    :rtype: TypeGuard[SyncSessionType]
    """
    return isinstance(value, SyncSessionType)


def is_sync_session_transaction(value: object) -> TypeGuard["SessionTransaction"]:
    """Check if a value is an SQLAlchemy sync session transaction.

    :param value: Value to check.
    :type value: object
    :returns: True if value is a sync session transaction.
    :rtype: TypeGuard[SessionTransaction]
    """
    return isinstance(value, SessionTransaction)


def is_str(value: object) -> TypeGuard[str]:
    """Check if a value is a string.

    :param value: Value to check.
    :type value: object
    :returns: True if value is a string.
    :rtype: TypeGuard[str]
    """
    return isinstance(value, str)


def is_list_of_str(value: object) -> TypeGuard[list[str]]:
    """Check if a value is a list containing only strings.

    :param value: Value to check.
    :type value: object
    :returns: True if value is a list of strings.
    :rtype: TypeGuard[list[str]]
    """
    return isinstance(value, list) and all(isinstance(s, str) for s in value)
