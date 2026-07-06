from typing import BinaryIO

import boto3
from botocore.config import Config

from src.configs import settings
from src.storage.base import Storage


class S3Storage(Storage):
    def __init__(self) -> None:
        session = boto3.Session(
            aws_access_key_id=settings.storage.access_key_id,
            aws_secret_access_key=settings.storage.secret_access_key,
        )
        self.client = session.client(
            "s3",
            endpoint_url=settings.storage.endpoint_url,
            region_name=settings.storage.region,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
            ),
        )
        self.bucket = settings.storage.bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_name,
            Body=data,
            ContentType=content_type,
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        object_name: str,
        content_type: str,
    ) -> None:
        self.client.upload_fileobj(
            Fileobj=stream,
            Bucket=self.bucket,
            Key=object_name,
            ExtraArgs={"ContentType": content_type},
        )

    def download_bytes(
        self,
        object_name: str,
    ) -> bytes:
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=object_name,
        )
        return response["Body"].read()

    def delete(
        self,
        object_name: str,
    ) -> None:
        self.client.delete_object(
            Bucket=self.bucket,
            Key=object_name,
        )
