from collections.abc import Callable, Coroutine
import logging
import asyncio
import random

import aio_pika

from src.configs import settings
from src.queue.queue import Queue

logger = logging.getLogger(__name__)


class RabbitMQQueue(Queue):
    def __init__(self) -> None:
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.Channel | None = None
        self._dlx_exchange: aio_pika.Exchange | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(
            host=settings.rabbitmq_host,
            port=settings.rabbitmq_port,
            login=settings.rabbitmq_user,
            password=settings.rabbitmq_password,
            virtualhost=settings.rabbitmq_vhost,
        )
        self._channel = await self._connection.channel()
        
        # Declare the Dead Letter Exchange (DLX)
        self._dlx_exchange = await self._channel.declare_exchange(
            "dlx",
            type=aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        
        # Pre-declare queues and their DLQs
        await self._setup_queue_with_dlq(settings.rabbitmq_queue)
        await self._setup_queue_with_dlq(settings.storage_queue)
        await self._setup_queue_with_dlq(settings.langgraph_queue)

    async def _setup_queue_with_dlq(self, queue_name: str) -> aio_pika.Queue:
        dlq_name = f"{queue_name}_dlq"
        
        # Declare the Dead Letter Queue
        dlq = await self._channel.declare_queue(dlq_name, durable=True)
        # Bind the DLQ to the DLX
        await dlq.bind(self._dlx_exchange, routing_key=dlq_name)
        
        try:
            # Declare the Main Queue with DLX configuration
            return await self._channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": dlq_name,
                },
            )
        except aio_pika.exceptions.ChannelPreconditionFailed as e:
            logger.warning(
                "Queue declaration failed for '%s' due to ChannelPreconditionFailed: %s. "
                "Re-creating channel, deleting the old queue, and redeclaring...",
                queue_name, e
            )
            # Recreate channel since PreconditionFailed closes it
            self._channel = await self._connection.channel()
            self._dlx_exchange = await self._channel.declare_exchange(
                "dlx",
                type=aio_pika.ExchangeType.DIRECT,
                durable=True,
            )
            # Redeclare DLQ and bind to DLX in the new channel
            dlq = await self._channel.declare_queue(dlq_name, durable=True)
            await dlq.bind(self._dlx_exchange, routing_key=dlq_name)
            
            # Delete conflicting queue
            await self._channel.queue_delete(queue_name)
            logger.info("Successfully deleted conflicting queue '%s'", queue_name)
            
            # Try declaring again
            return await self._channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": dlq_name,
                },
            )


    async def publish(
        self,
        queue_name: str,
        message: bytes,
        headers: dict | None = None,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("Queue not connected. Call connect() first.")
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=message,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=headers or {},
            ),
            routing_key=queue_name,
        )

    async def consume(
        self,
        queue_name: str,
        callback: Callable[[bytes], Coroutine],
        max_retries: int = 3,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("Queue not connected. Call connect() first.")
        
        queue = await self._setup_queue_with_dlq(queue_name)

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            headers = dict(message.headers or {})
            retry_count = headers.get("x-retry-count", 0)
            
            try:
                await callback(message.body)
                await message.ack()
            except Exception as e:
                logger.exception("Error processing message on queue %s: %s", queue_name, e)
                
                if retry_count < max_retries:
                    new_headers = {**headers, "x-retry-count": retry_count + 1}
                    # Exponential backoff with jitter: min(2 * 2^n, 30) + uniform(0, 1)
                    backoff = min(2.0 * (2 ** retry_count), 30.0) + random.uniform(0, 1)
                    logger.warning(
                        "Retrying message on queue %s (%d/%d) in %.1fs",
                        queue_name, retry_count + 1, max_retries, backoff
                    )
                    await asyncio.sleep(backoff)
                    
                    await self.publish(
                        queue_name=queue_name,
                        message=message.body,
                        headers=new_headers,
                    )
                    await message.ack()
                else:
                    logger.error(
                        "Message on queue %s exceeded max retries (%d). Rejecting to DLQ.",
                        queue_name, max_retries
                    )
                    await message.reject(requeue=False)

        await queue.consume(on_message)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._dlx_exchange = None

