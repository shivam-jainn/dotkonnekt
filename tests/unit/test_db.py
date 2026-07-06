from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.db import DatabasePool


@pytest.mark.unit
class TestDatabasePool:
    @patch("src.database.db.asyncpg")
    async def test_connect_creates_pool_and_inits_schema(self, mock_asyncpg):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        db = DatabasePool()
        result = await db.connect()

        mock_asyncpg.create_pool.assert_awaited_once()
        assert result is mock_pool
        assert db._pool is mock_pool
        mock_conn.execute.assert_awaited_once()

    @patch("src.database.db.asyncpg")
    async def test_connect_is_idempotent(self, mock_asyncpg):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        db = DatabasePool()
        await db.connect()
        result = await db.connect()

        mock_asyncpg.create_pool.assert_awaited_once()
        assert result is mock_pool

    @patch("src.database.db.asyncpg")
    async def test_close_resets_pool(self, mock_asyncpg):
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        db = DatabasePool()
        await db.connect()
        await db.close()

        mock_pool.close.assert_awaited_once()
        assert db._pool is None

    @patch("src.database.db.asyncpg")
    async def test_close_when_not_connected_does_nothing(self, mock_asyncpg):
        db = DatabasePool()
        await db.close()

        mock_asyncpg.create_pool.assert_not_called()
        assert db._pool is None

    async def test_pool_raises_when_not_initialised(self):
        db = DatabasePool()
        with pytest.raises(RuntimeError, match="not initialised"):
            _ = db.pool

    @patch("src.database.db.asyncpg")
    async def test_pool_returns_after_connect(self, mock_asyncpg):
        mock_pool = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        db = DatabasePool()
        await db.connect()
        assert db.pool is mock_pool

    @patch("src.database.db.asyncpg")
    async def test_init_schema_executes_create_table(self, mock_asyncpg):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        db = DatabasePool()
        await db.connect()

        mock_conn.execute.assert_awaited_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS jobs" in sql
