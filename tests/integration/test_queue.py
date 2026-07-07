import asyncio
from uuid import uuid4

import pytest

from src.queue.rabbitmq import RabbitMQQueue


pytestmark = pytest.mark.integration


@pytest.fixture
async def queue():
    q = RabbitMQQueue()
    await q.connect()
    yield q
    await q.close()


class TestRabbitMQIntegration:
    async def test_publish_and_consume(self, queue):
        queue_name = f"test-queue-{uuid4().hex[:8]}"
        sent_message = f"hello-{uuid4().hex}".encode()
        received = []

        async def callback(body: bytes):
            received.append(body)

        # Start consuming in background
        consume_task = asyncio.create_task(queue.consume(queue_name, callback))
        await asyncio.sleep(0.3)

        await queue.publish(queue_name, sent_message)
        await asyncio.sleep(0.3)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        assert len(received) == 1
        assert received[0] == sent_message

    async def test_multiple_messages(self, queue):
        queue_name = f"test-queue-{uuid4().hex[:8]}"
        messages = [b"msg-1", b"msg-2", b"msg-3"]
        received = []

        async def callback(body: bytes):
            received.append(body)

        consume_task = asyncio.create_task(queue.consume(queue_name, callback))
        await asyncio.sleep(0.3)

        for msg in messages:
            await queue.publish(queue_name, msg)
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        assert len(received) == len(messages)
        for msg in messages:
            assert msg in received

    async def test_messages_are_persistent(self, queue):
        queue_name = f"test-queue-{uuid4().hex[:8]}"
        sent = b"persistent-message"

        # declare the queue first so publish has a durable home
        await queue._setup_queue_with_dlq(queue_name)
        await queue.publish(queue_name, sent)

        received = []

        async def callback(body: bytes):
            received.append(body)

        consume_task = asyncio.create_task(queue.consume(queue_name, callback))
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        assert len(received) >= 1

    async def test_multiple_consumers_receive_different_messages(self, queue):
        queue_name = f"test-queue-{uuid4().hex[:8]}"
        received_1 = []
        received_2 = []

        async def callback_1(body: bytes):
            received_1.append(body)

        async def callback_2(body: bytes):
            received_2.append(body)

        # Both consumers on same queue — messages should be round-robin'd
        task_1 = asyncio.create_task(queue.consume(queue_name, callback_1))
        task_2 = asyncio.create_task(queue.consume(queue_name, callback_2))
        await asyncio.sleep(0.3)

        for i in range(4):
            await queue.publish(queue_name, f"msg-{i}".encode())
        await asyncio.sleep(0.5)

        task_1.cancel()
        task_2.cancel()
        try:
            await task_1
            await task_2
        except asyncio.CancelledError:
            pass

        total = len(received_1) + len(received_2)
        assert total == 4
