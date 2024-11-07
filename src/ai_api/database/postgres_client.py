from sqlalchemy import create_engine, text
from sqlalchemy.exc import InvalidRequestError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session, sessionmaker, SessionTransaction

from ai_api.util.guard import is_async_session, is_sync_session, is_sync_session_transaction


class DatabaseBase:

    def __init__(self, url: str = "sqlite://"):
        self.url = url
        self.init_extension()

    def init_extension(self):
        engine = create_engine(self.url.replace("+asyncpg", ""))
        if "sqlite" in self.url:
            return

        conn = engine.connect()
        with Session(conn) as session:
            statement = text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
            session.execute(statement)
            session.commit()

        conn.close()

    def init_diskann(self):
        engine = create_engine(self.url.replace("+asyncpg", ""))
        conn = engine.connect()

        with Session(conn) as session:
            statement = text("CREATE INDEX IF NOT EXISTS document_embedding_idx ON entry USING diskann (vector);")
            session.execute(statement)
            session.commit()

        conn.close()


class AIOPostgres(DatabaseBase):
    """
    An async context manager for the postgres database.
    """

    def __init__(self, url: str = "sqlite+aiosqlite://"):
        super().__init__(url)
        self.engine = create_async_engine(self.url)
        self._session = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

        self._async_context: AsyncSessionType | None = None
        self._transaction: AsyncSessionType | None = None

    async def __aenter__(self) -> AsyncSessionType:
        self._async_context = self._session()
        self._transaction = await self._async_context.__aenter__()
        return self._transaction

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not is_async_session(self._transaction) or not is_async_session(self._async_context):
            raise ValueError("Postgres client is not in a valid state")

        try:
            await self._transaction.commit()
        except InvalidRequestError | SQLAlchemyError as e:
            await self._transaction.rollback()
            raise e
        finally:
            await self._async_context.__aexit__(exc_type, exc_val, exc_tb)
            self._async_context = None
            self._transaction = None


class Postgres(DatabaseBase):
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
        self._transaction = self._context.begin()
        return self._context

    def __exit__(self, exc_type, exc_val, exc_tb):
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
