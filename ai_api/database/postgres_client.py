"""Module providing PostgreSQL database client implementations.

This module provides both synchronous and asynchronous PostgreSQL client implementations
with support for connection pooling, transaction management, and vector operations.
The clients support both regular PostgreSQL operations and vector-specific extensions
like vectorscale and pgvector for similarity search operations.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Self

from sqlalchemy import create_engine, text
from sqlalchemy.exc import InvalidRequestError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm.session import Session, SessionTransaction, sessionmaker

from ai_api import settings
from ai_api.util.logger import get_logger
from ai_api.util.singleton import Singleton

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from types import EllipsisType, TracebackType

    from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType


class DatabaseBase:
    """Base class for database client implementations.

    This class provides common functionality for database initialization,
    including vector extension setup for similarity search operations.

    :param url: Database connection URL.
    :type url: str
    """

    def __init__(self, url: str = "sqlite://") -> None:
        """Initialize the database base and initialize the extension.

        :param url: Database connection URL.
        :type url: str
        """
        self.url = url
        self.logger = get_logger()
        self.init_extension()

    def init_extension(self) -> None:
        """Initialize database extensions for vector operations.

        Attempts to initialize vectorscale first, falling back to pgvector.
        This operation is skipped in debug environments.

        :raises: May raise SQLAlchemy errors during extension initialization.
        """
        if settings.debug:
            return

        engine = create_engine(self.url.replace("+asyncpg", ""))

        if "sqlite" in self.url:
            return

        conn = engine.connect()
        with self.logger.catch(message="Failed to initialize vectorscale"), Session(conn) as session:
            statement = text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
            session.execute(statement)
            session.commit()

        with self.logger.catch(message="Failed to initialize pgvector"), Session(conn) as session:
            statement = text("CREATE EXTENSION IF NOT EXISTS vector;")
            session.execute(statement)
            session.commit()

        conn.close()

    def init_diskann(self) -> None:
        """Initialize vector similarity search indices.

        Attempts to initialize diskann index if vectorscale is available,
        otherwise falls back to hnsw from pgvector.

        :raises: May raise SQLAlchemy errors during index creation.
        """
        engine = create_engine(self.url.replace("+asyncpg", ""))
        conn = engine.connect()

        with Session(conn) as session:
            check_stmt = text("SELECT extname FROM pg_extension WHERE extname = 'vectorscale';")
            result = session.execute(check_stmt)
            if result.scalar():
                with self.logger.catch(message="Failed to initialize diskann"):
                    statement = text("CREATE INDEX IF NOT EXISTS document_embedding_idx ON entry USING diskann (vector vector_ip_ops);")
                    session.execute(statement)
                    session.commit()
            else:
                with self.logger.catch(message="Failed to initialize hnsw"):
                    statement = text("CREATE INDEX IF NOT EXISTS document_embedding_idx ON entry USING hnsw (vector vector_ip_ops);")
                    session.execute(statement)
                    session.commit()

        conn.close()


class AIOPostgresBase(DatabaseBase, metaclass=Singleton):
    """Base class for asynchronous PostgreSQL client implementation.

    This class provides asynchronous database operations with connection pooling
    and transaction management using SQLAlchemy's async engine.

    :param url: Database connection URL.
    :type url: str
    """

    _initialized = False
    _context_var: ContextVar[AsyncSessionType | None] = ContextVar("postgres_context", default=None)
    _transaction_var: ContextVar[AsyncSessionType | None] = ContextVar("postgres_transaction", default=None)

    def __init__(self, url: str = "sqlite+aiosqlite://") -> None:
        """Initialize the async PostgreSQL client with connection pooling.

        :param url: Database connection URL.
        :type url: str
        """
        if not self._initialized:
            super().__init__(url)
            self.engine = create_async_engine(
                self.url,
                pool_size=20,
                max_overflow=10,
                pool_timeout=15,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False,
                pool_use_lifo=True,
                connect_args={
                    "server_settings": {
                        "application_name": "gallery-ai",
                        "tcp_keepalives_idle": "60",
                        "tcp_keepalives_interval": "10",
                        "tcp_keepalives_count": "6",
                        "statement_timeout": "29000",
                        "idle_in_transaction_session_timeout": "29000",
                    },
                },
            )
            self._session = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            self._initialized = True

    async def check_connection(self) -> None:
        """Check if database connection is healthy.

        :raises: SQLAlchemyError if connection check fails
        """
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except SQLAlchemyError:
            self.logger.exception("Database connection check failed")
            raise

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSessionType]:
        """Create an async session context with connection validation.

        :yields: Async database session.
        :rtype: AsyncIterator[AsyncSessionType]
        :raises: SQLAlchemyError if connection check fails
        """
        await self.check_connection()

        async_session: AsyncSessionType = self._session()
        try:
            yield async_session
        finally:
            await async_session.close()

    async def __aenter__(self) -> AsyncSessionType:
        """Create an async transaction context.

        :returns: Async session with transaction.
        :rtype: AsyncSessionType
        """
        async_context: AsyncSessionType = self._session()
        self._context_var.set(async_context)

        transaction = await async_context.__aenter__()
        self._transaction_var.set(transaction)
        return transaction

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        """Exit the async transaction context.

        :param exc_type: Exception type if an error occurred.
        :type exc_type: type[BaseException] | None
        :param exc_val: Exception value if an error occurred.
        :type exc_val: BaseException | None
        :param exc_tb: Exception traceback if an error occurred.
        :type exc_tb: TracebackType | None
        :raises: Re-raises any SQLAlchemy errors after rollback.
        """
        async_context = self._context_var.get()
        transaction = self._transaction_var.get()

        if transaction is None or async_context is None:
            return

        try:
            await transaction.commit()
        except InvalidRequestError:
            pass
        except SQLAlchemyError:
            await transaction.rollback()
            raise
        finally:
            await async_context.__aexit__(exc_type, exc_val, exc_tb)
            self._context_var.set(None)
            self._transaction_var.set(None)


class AIOPostgres(AIOPostgresBase):
    """Async PostgreSQL client with debug environment handling.

    This class extends AIOPostgresBase with special handling for debug environments,
    where database operations are mocked.

    :param url: Database connection URL.
    :type url: str
    """

    def __init__(self, url: str = settings.postgres.url) -> None:
        """Initialize the async PostgreSQL client.

        :param url: Database connection URL.
        :type url: str
        """
        if settings.debug:
            return
        super().__init__(url)

    def __getattr__(self, name: str) -> Callable[[Any], None | EllipsisType]:
        """Get attribute with debug environment handling.

        :param name: Attribute name.
        :type name: str
        :returns: Mock function in debug, actual attribute otherwise.
        :rtype: Callable[[Any], None | EllipsisType]
        """
        if settings.debug:
            return lambda *_, **__: ...
        return getattr(super(AIOPostgresBase, self), name)


class PostgresBase(DatabaseBase):
    """Base class for synchronous PostgreSQL client implementation.

    This class provides synchronous database operations with transaction
    management using SQLAlchemy's sync engine.

    :param url: Database connection URL.
    :type url: str
    :param init_diskann: Whether to initialize vector similarity indices.
    :type init_diskann: bool
    """

    def __init__(self, url: str = "sqlite://", init_diskann: bool = False) -> None:  # noqa: FBT001, FBT002
        """Initialize the synchronous PostgreSQL client.

        :param url: Database connection URL.
        :type url: str
        :param init_diskann: Whether to initialize vector similarity indices.
        :type init_diskann: bool
        """
        super().__init__(url)
        self.engine = create_engine(self.url.replace("+asyncpg", ""))
        self._session = sessionmaker(bind=self.engine)

        if init_diskann:
            self.init_diskann()

        self._context: Session | None = None
        self._transaction: SessionTransaction | None = None

    def __enter__(self) -> Session:
        """Create a synchronous transaction context.

        :returns: Database session with transaction.
        :rtype: Session
        :raises ValueError: If client is not in a valid state.
        """
        self._context = self._session()
        if not self._context:
            self.logger.exception("Postgres client is not in a valid state")
            raise ValueError
        self._transaction = self._context.begin()
        return self._context

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        """Exit the synchronous transaction context.

        :param exc_type: Exception type if an error occurred.
        :type exc_type: type[BaseException] | None
        :param exc_val: Exception value if an error occurred.
        :type exc_val: BaseException | None
        :param exc_tb: Exception traceback if an error occurred.
        :type exc_tb: TracebackType | None
        :raises ValueError: If client is not in a valid state.
        :raises: Re-raises any SQLAlchemy errors after rollback.
        """


class Postgres(PostgresBase):
    """Synchronous PostgreSQL client with debug environment handling.

    This class extends PostgresBase with special handling for debug environments,
    where database operations are mocked.

    :param url: Database connection URL.
    :type url: str
    :param init_diskann: Whether to initialize vector similarity indices.
    :type init_diskann: bool
    """

    def __init__(self, url: str = settings.postgres.url.replace("+asyncpg", ""), init_diskann: bool = False) -> None:  # noqa: FBT001, FBT002
        """Initialize the synchronous PostgreSQL client.

        :param url: Database connection URL.
        :type url: str
        :param init_diskann: Whether to initialize vector similarity indices.
        :type init_diskann: bool
        """
        if settings.debug:
            return
        super().__init__(url, init_diskann)

    def __getattr__(self, name: str) -> Callable[[Any], None | EllipsisType]:
        """Get attribute with debug environment handling.

        :param name: Attribute name.
        :type name: str
        :returns: Mock function in debug, actual attribute otherwise.
        :rtype: Callable[[Any], None | EllipsisType]
        """
        if settings.debug:
            return lambda *_, **__: ...
        return getattr(super(PostgresBase, self), name)


class MockAIOPostgres:
    """Mock AIOPostgres class for debug environment."""

    def __init__(self, *_: Any, **__: Any) -> None:
        """Initialize the mock AIOPostgres class."""
        get_logger().warning("Mock AIOPostgres class is being used.")

    async def __aenter__(self) -> Self:
        """Enter the mock AIOPostgres context."""
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        """Exit the mock AIOPostgres context."""

    def __await__(self) -> Self:
        """Return the mock AIOPostgres context."""
        return self

    def session(self) -> Self:
        """Return the mock AIOPostgres context."""
        return self

    def __getattribute__(self, name: str) -> Self | Any:
        """Get attribute with debug environment handling."""
        if name in ("__aenter__", "__aexit__", "__await__", "session"):
            return super().__getattribute__(name)
        return self


if settings.debug:
    AIOPostgres = MockAIOPostgres  # type: ignore[]
