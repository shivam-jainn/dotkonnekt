from abc import ABC, abstractmethod
from typing import Any


class Storage(ABC):
    @abstractmethod
    async def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
    ) -> None: ...

    @abstractmethod
    async def upload_stream(
        self,
        stream: Any,
        object_name: str,
        content_type: str,
    ) -> int: ...

    @abstractmethod
    async def download_bytes(
        self,
        object_name: str,
    ) -> bytes: ...

    @abstractmethod
    async def delete(
        self,
        object_name: str,
    ) -> None: ...

