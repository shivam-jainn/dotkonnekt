from src.queue.queue import Queue
from src.queue.rabbitmq import RabbitMQQueue

queue: Queue = RabbitMQQueue()

__all__ = [
    "Queue",
    "RabbitMQQueue",
    "queue",
]
