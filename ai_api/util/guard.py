from typing import Any, TypeGuard

from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session as SyncSessionType
from sqlalchemy.orm.session import SessionTransaction


def is_async_session(session: Any) -> TypeGuard["AsyncSessionType"]:
    return isinstance(session, AsyncSessionType)


def is_sync_session(session: Any) -> TypeGuard["SyncSessionType"]:
    return isinstance(session, SyncSessionType)


def is_sync_session_transaction(session: Any) -> TypeGuard["SessionTransaction"]:
    return isinstance(session, SessionTransaction)


def is_str(s: Any) -> TypeGuard[str]:
    return isinstance(s, str)


def is_list_of_str(l: Any) -> TypeGuard[list[str]]:
    return isinstance(l, list) and all(isinstance(s, str) for s in l)
