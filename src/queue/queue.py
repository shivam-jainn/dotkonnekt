from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine


class Queue(ABC):
    @abstractmethod
    async def publish(self, queue_name: str, message: bytes) -> None:
        ...

    @abstractmethod
    async def consume(
        self,
        queue_name: str,
        callback: Callable[[bytes], Coroutine],
    ) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
