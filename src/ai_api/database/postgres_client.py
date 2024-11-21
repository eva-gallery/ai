import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import InvalidRequestError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session, sessionmaker, SessionTransaction

from ai_api.util.guard import is_sync_session, is_sync_session_transaction
from ai_api.util.singleton import Singleton


class DatabaseBase:
    def __init__(self, url: str = "sqlite://"):
        self.url = url
        self.init_extension()

    def init_extension(self) -> None:
        if os.getenv("CI"):
            return

        engine = create_engine(self.url.replace("+asyncpg", ""))
        
        if "sqlite" in self.url:
            return

        conn = engine.connect()
        try:
            with Session(conn) as session:
                statement = text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
                session.execute(statement)
                session.commit()
        except Exception:
            try:
                with Session(conn) as session:
                    statement = text("CREATE EXTENSION IF NOT EXISTS vector;")
                    session.execute(statement)
                    session.commit()
            except Exception as e2:
                raise e2

        conn.close()

    def init_diskann(self) -> None:
        engine = create_engine(self.url.replace("+asyncpg", ""))
        conn = engine.connect()

        with Session(conn) as session:
            check_stmt = text("SELECT extname FROM pg_extension WHERE extname = 'vectorscale';")
            result = session.execute(check_stmt)
            if result.scalar():
                try:
                    statement = text("CREATE INDEX IF NOT EXISTS document_embedding_idx ON entry USING diskann (vector vector_ip_ops);")
                    session.execute(statement)
                    session.commit()
                except Exception:
                    pass
            else:
                try:
                    statement = text("CREATE INDEX IF NOT EXISTS document_embedding_idx ON entry USING hnsw (vector vector_ip_ops);")
                    session.execute(statement)
                    session.commit()
                except Exception:
                    pass
        conn.close()


class AIOPostgresBase(DatabaseBase, metaclass=Singleton):
    """
    An async context manager for the postgres database.
    """
    _initialized = False
    _context_var: ContextVar[AsyncSessionType | None] = ContextVar('postgres_context', default=None)
    _transaction_var: ContextVar[AsyncSessionType | None] = ContextVar('postgres_transaction', default=None)

    def __init__(self, url: str = "sqlite+aiosqlite://"):
        if not self._initialized:
            super().__init__(url)
            self.engine = create_async_engine(
                self.url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True
            )
            self._session = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            self._initialized = True

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSessionType]:
        """Get a session context."""
        async_session: AsyncSessionType = self._session()
        try:
            yield async_session
        finally:
            await async_session.close()

    async def __aenter__(self) -> AsyncSessionType:
        """Creates a transactional context for the database."""
        async_context: AsyncSessionType = self._session()
        self._context_var.set(async_context)

        transaction = await async_context.__aenter__()
        self._transaction_var.set(transaction)
        return transaction

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        async_context = self._context_var.get()
        transaction = self._transaction_var.get()

        if transaction is None or async_context is None:
            return

        try:
            await transaction.commit()
        except InvalidRequestError:
            pass
        except SQLAlchemyError as e:
            await transaction.rollback()
            raise e
        finally:
            await async_context.__aexit__(exc_type, exc_val, exc_tb)
            self._context_var.set(None)
            self._transaction_var.set(None)


class AIOPostgres(AIOPostgresBase):
    def __init__(self, url: str = "sqlite+aiosqlite://"):
        if os.getenv("CI"):
            return
        super().__init__(url)
    
    def __getattr__(self, name: str) -> Any:
        if os.getenv("CI"):
            return lambda *args, **kwargs: ...
        return getattr(super(AIOPostgresBase, self), name)

class PostgresBase(DatabaseBase):
    """
    The synchronous version of the AIOPostgres class.
    """

    def __init__(self, url: str = "sqlite://", init_diskann: bool = False):
        super().__init__(url)
        self.engine = create_engine(self.url.replace("+asyncpg", ""))
        self._session = sessionmaker(bind=self.engine)

        if init_diskann:
            self.init_diskann()

        self._context: Session | None = None
        self._transaction: SessionTransaction | None = None

    def __enter__(self) -> Session:
        self._context = self._session()
        if not self._context:
            raise ValueError("Postgres client is not in a valid state")
        self._transaction = self._context.begin()
        return self._context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not is_sync_session_transaction(self._transaction) or not is_sync_session(self._context):
            raise ValueError("Postgres client is not in a valid state")

        try:
            self._transaction.commit()
        except SQLAlchemyError as e:
            self._transaction.rollback()
            raise e
        finally:
            self._context.close()
            self._context = None
            self._transaction = None

class Postgres(PostgresBase):
    def __init__(self, url: str = "sqlite://", init_diskann: bool = False):
        if os.getenv("CI"):
            return
        super().__init__(url, init_diskann)

    def __getattr__(self, name: str) -> Any:
        if os.getenv("CI"):
            return lambda *args, **kwargs: ...
        return getattr(super(PostgresBase, self), name)
