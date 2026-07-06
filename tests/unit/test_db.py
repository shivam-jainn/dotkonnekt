from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.db import DatabaseManager


@pytest.mark.unit
class TestDatabaseManager:
    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_connect_creates_engine_and_session_factory(
        self, mock_sessionmaker, mock_create_engine
    ):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        db = DatabaseManager()
        await db.connect()

        mock_create_engine.assert_called_once()
        mock_conn.execute.assert_awaited_once()
        assert db._engine is mock_engine
        assert db._session_factory is mock_session_factory

    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_connect_is_idempotent(self, mock_sessionmaker, mock_create_engine):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()

        db = DatabaseManager()
        await db.connect()
        await db.connect()

        mock_create_engine.assert_called_once()

    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_close_disposes_engine(self, mock_sessionmaker, mock_create_engine):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()

        db = DatabaseManager()
        await db.connect()
        await db.close()

        mock_engine.dispose.assert_awaited_once()
        assert db._engine is None
        assert db._session_factory is None

    @patch("src.database.db.create_engine")
    async def test_close_when_not_connected_does_nothing(self, mock_create_engine):
        db = DatabaseManager()
        await db.close()
        mock_create_engine.assert_not_called()

    async def test_pool_raises_when_not_initialised(self):
        db = DatabaseManager()
        with pytest.raises(RuntimeError, match="not initialised"):
            _ = db.pool

    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_pool_returns_after_connect(
        self, mock_sessionmaker, mock_create_engine
    ):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        db = DatabaseManager()
        await db.connect()
        assert db.pool is mock_session_factory

    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_health_check_returns_true(
        self, mock_sessionmaker, mock_create_engine
    ):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()

        db = DatabaseManager()
        await db.connect()
        result = await db.health_check()

        assert result is True

    @patch("src.database.db.create_engine")
    @patch("src.database.db.async_sessionmaker")
    async def test_health_check_returns_false_on_failure(
        self, mock_sessionmaker, mock_create_engine
    ):
        mock_engine = MagicMock()
        connect_calls = 0

        async def connect_cm():
            nonlocal connect_calls
            mock_conn = AsyncMock()
            if connect_calls == 0:
                mock_conn.execute = AsyncMock()
            else:
                mock_conn.execute = AsyncMock(side_effect=Exception("db error"))
            connect_calls += 1
            return mock_conn

        mock_engine.connect.return_value.__aenter__ = AsyncMock(side_effect=connect_cm)
        mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()

        db = DatabaseManager()
        await db.connect()
        result = await db.health_check()

        assert result is False

    @patch("src.database.db.create_engine")
    async def test_connect_retries_on_failure(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection failed")
        mock_create_engine.return_value = mock_engine

        db = DatabaseManager()
        with pytest.raises(
            ConnectionError, match="Could not connect to database after 2 attempts"
        ):
            await db.connect(retry_attempts=2, retry_delay=0.01)

        assert mock_create_engine.call_count == 2
