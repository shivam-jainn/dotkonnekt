from collections.abc import Callable, Coroutine

import aio_pika

from src.configs import settings
from src.queue.queue import Queue


class RabbitMQQueue(Queue):
    def __init__(self) -> None:
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.Channel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(
            host=settings.db.rabbitmq_host,
            port=settings.db.rabbitmq_port,
            login=settings.db.rabbitmq_user,
            password=settings.db.rabbitmq_password,
            virtualhost=settings.db.rabbitmq_vhost,
        )
        self._channel = await self._connection.channel()
        await self._channel.declare_queue(
            settings.db.rabbitmq_queue,
            durable=True,
        )

    async def publish(self, queue_name: str, message: bytes) -> None:
        if self._channel is None:
            raise RuntimeError("Queue not connected. Call connect() first.")
        await self._channel.default_exchange.publish(
            aio_pika.Message(body=message, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=queue_name,
        )

    async def consume(
        self,
        queue_name: str,
        callback: Callable[[bytes], Coroutine],
    ) -> None:
        if self._channel is None:
            raise RuntimeError("Queue not connected. Call connect() first.")
        queue = await self._channel.declare_queue(queue_name, durable=True)

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            async with message.process():
                await callback(message.body)

        await queue.consume(on_message)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._channel = None
