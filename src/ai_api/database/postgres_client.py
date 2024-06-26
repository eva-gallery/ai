from sqlalchemy import create_engine, text
from sqlalchemy.exc import InvalidRequestError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio.session import AsyncSession as AsyncSessionType
from sqlalchemy.orm.session import Session


class AIOPostgres:
    """
    An async context manager for the postgres database.
    """

    def __init__(self, url: str = None):
        self.url = url
        self.init_extension()

        self._engine = create_async_engine(self.url)
        self._session = async_sessionmaker(bind=self._engine, class_=AsyncSession, expire_on_commit=False)

        self._async_context: AsyncSessionType | None = None
        self._transaction: AsyncSessionType | None = None

    def init_extension(self):
        engine = create_engine(self.url.replace("+asyncpg", ""))
        conn = engine.connect()

        with Session(conn) as session:
            statement = text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
            session.execute(statement)
            session.commit()

        conn.close()

    async def __aenter__(self) -> AsyncSessionType:
        self._async_context = self._session()
        self._transaction = await self._async_context.__aenter__()
        return self._transaction

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._transaction.commit()
        except InvalidRequestError | SQLAlchemyError as e:
            await self._transaction.rollback()
            raise e
        finally:
            await self._async_context.__aexit__(exc_type, exc_val, exc_tb)
            self._async_context = None
            self._transaction = None
