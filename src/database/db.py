import asyncpg

from src.configs import settings


CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'queued',
    files       JSONB NOT NULL DEFAULT '[]',
    collection  TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class DatabasePool:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=settings.db.postgres_host,
                port=settings.db.postgres_port,
                database=settings.db.postgres_database,
                user=settings.db.postgres_user,
                password=settings.db.postgres_password,
                min_size=settings.db.pool_min_size,
                max_size=settings.db.pool_max_size,
            )
            await self._init_schema()
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialised. Call connect() first.")
        return self._pool

    async def _init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_JOBS_TABLE)


db = DatabasePool()
