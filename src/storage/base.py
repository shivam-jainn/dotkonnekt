from abc import ABC, abstractmethod
from typing import BinaryIO


class Storage(ABC):
    @abstractmethod
    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
    ) -> None:
        ...

    @abstractmethod
    def upload_stream(
        self,
        stream: BinaryIO,
        object_name: str,
        content_type: str,
    ) -> None:
        ...

    @abstractmethod
    def download_bytes(
        self,
        object_name: str,
    ) -> bytes:
        ...

    @abstractmethod
    def delete(
        self,
        object_name: str,
    ) -> None:
        ...
