import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.configs import settings
from src.database.models import Base


def build_database_url() -> str:
    if settings.database_url:
        return settings.database_url
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def create_engine() -> AsyncEngine:
    return create_async_engine(
        build_database_url(),
        pool_size=settings.pool_min_size,
        max_overflow=settings.pool_max_size - settings.pool_min_size,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        connect_args={
            "command_timeout": 30,
        },
    )


class DatabaseManager:
    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(
        self,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        if self._engine is not None:
            return
        last_exc: Exception | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                engine = create_engine()
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                self._engine = engine
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < retry_attempts:
                    await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))

        raise ConnectionError(
            f"Could not connect to database after {retry_attempts} attempts"
        ) from last_exc

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @property
    def pool(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialised. Call connect() first.")
        return self._session_factory

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database not initialised. Call connect() first.")
        return self._engine

    async def create_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def health_check(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


db = DatabaseManager()


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    async with db.pool() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
