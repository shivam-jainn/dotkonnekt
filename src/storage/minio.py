import io
from typing import BinaryIO

from minio import Minio

from src.configs import settings
from src.storage.base import Storage


class MinioStorage(Storage):
    def __init__(self) -> None:
        endpoint = (
            settings.storage_endpoint_url
            .replace("http://", "")
            .replace("https://", "")
        )
        self.client = Minio(
            endpoint=endpoint,
            access_key=settings.storage_access_key_id,
            secret_key=settings.storage_secret_access_key,
            secure=settings.storage_use_ssl,
        )
        self.bucket = settings.storage_bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
    ) -> None:
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        object_name: str,
        content_type: str,
    ) -> None:
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=stream,
            length=-1,
            content_type=content_type,
            part_size=10 * 1024 * 1024,
        )

    def download_bytes(
        self,
        object_name: str,
    ) -> bytes:
        response = self.client.get_object(
            bucket_name=self.bucket,
            object_name=object_name,
        )
        return response.read()

    def delete(
        self,
        object_name: str,
    ) -> None:
        self.client.remove_object(
            bucket_name=self.bucket,
            object_name=object_name,
        )
