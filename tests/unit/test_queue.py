from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.queue.rabbitmq import RabbitMQQueue


@pytest.mark.unit
class TestRabbitMQQueue:
    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_connect_opens_connection_and_channel_and_declares_queue(self, mock_connect):
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel
        mock_queue = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue

        queue = RabbitMQQueue()
        await queue.connect()

        mock_connect.assert_awaited_once()
        mock_conn.channel.assert_awaited_once()
        mock_channel.declare_queue.assert_awaited_once_with("ingestion", durable=True)
        assert queue._connection is mock_conn
        assert queue._channel is mock_channel

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_connect_is_idempotent(self, mock_connect):
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel

        queue = RabbitMQQueue()
        await queue.connect()
        mock_connect.reset_mock()
        mock_conn.channel.reset_mock()

        mock_conn2 = AsyncMock()
        mock_channel2 = AsyncMock()
        mock_connect.return_value = mock_conn2
        mock_conn2.channel.return_value = mock_channel2

        await queue.connect()
        mock_connect.assert_awaited_once()

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_publish_sends_message_to_exchange(self, mock_connect):
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel

        queue = RabbitMQQueue()
        await queue.connect()

        await queue.publish("test-queue", b"hello world")

        mock_channel.default_exchange.publish.assert_awaited_once()
        call_args = mock_channel.default_exchange.publish.call_args
        message = call_args[0][0]
        assert message.body == b"hello world"
        assert call_args[1]["routing_key"] == "test-queue"

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_publish_raises_when_not_connected(self, mock_connect):
        queue = RabbitMQQueue()
        with pytest.raises(RuntimeError, match="not connected"):
            await queue.publish("q", b"data")

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_consume_declares_queue_and_starts_consuming(self, mock_connect):
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel
        mock_queue_obj = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue_obj

        queue = RabbitMQQueue()
        await queue.connect()

        async def dummy_callback(body: bytes):
            pass

        await queue.consume("test-queue", dummy_callback)

        mock_channel.declare_queue.assert_awaited_with("test-queue", durable=True)
        mock_queue_obj.consume.assert_called_once()

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_consume_raises_when_not_connected(self, mock_connect):
        queue = RabbitMQQueue()

        async def dummy_callback(body: bytes):
            pass

        with pytest.raises(RuntimeError, match="not connected"):
            await queue.consume("q", dummy_callback)

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_close_closes_connection_and_resets_state(self, mock_connect):
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel

        queue = RabbitMQQueue()
        await queue.connect()
        await queue.close()

        mock_conn.close.assert_awaited_once()
        assert queue._connection is None
        assert queue._channel is None

    @patch("src.queue.rabbitmq.aio_pika.connect_robust")
    async def test_close_when_not_connected_does_nothing(self, mock_connect):
        queue = RabbitMQQueue()
        await queue.close()

        mock_connect.assert_not_called()
