import logging
from typing import Any
import aioboto3
from botocore.client import Config

from src.configs import settings
from src.storage.base import Storage

logger = logging.getLogger(__name__)


class S3Storage(Storage):
    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.bucket = settings.storage_bucket_name
        self.endpoint_url = settings.storage_endpoint_url
        self.aws_access_key_id = settings.storage_access_key_id
        self.aws_secret_access_key = settings.storage_secret_access_key
        self.region_name = settings.storage_region
        self.use_ssl = settings.storage_use_ssl

    def _get_client_args(self) -> dict:
        return {
            "service_name": "s3",
            "endpoint_url": self.endpoint_url,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.region_name,
            "use_ssl": self.use_ssl,
            "config": Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
            ),
        }

    async def _ensure_bucket(self, client) -> None:
        try:
            await client.head_bucket(Bucket=self.bucket)
        except Exception:
            logger.info("Bucket %s does not exist, creating it...", self.bucket)
            await client.create_bucket(Bucket=self.bucket)

    async def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
    ) -> None:
        async with self.session.client(**self._get_client_args()) as client:
            await self._ensure_bucket(client)
            await client.put_object(
                Bucket=self.bucket,
                Key=object_name,
                Body=data,
                ContentType=content_type,
            )

    async def upload_stream(
        self,
        stream: Any,
        object_name: str,
        content_type: str,
    ) -> int:
        async with self.session.client(**self._get_client_args()) as client:
            await self._ensure_bucket(client)
            
            mpu = await client.create_multipart_upload(
                Bucket=self.bucket,
                Key=object_name,
                ContentType=content_type,
            )
            upload_id = mpu["UploadId"]
            parts = []
            part_number = 1
            total_bytes = 0
            
            try:
                chunk_size = 5 * 1024 * 1024
                
                while True:
                    if hasattr(stream, "read"):
                        import inspect
                        res = stream.read(chunk_size)
                        if inspect.iscoroutine(res):
                            chunk = await res
                        else:
                            chunk = res
                    else:
                        chunk = b""
                        async for b in stream:
                            chunk += b
                            if len(chunk) >= chunk_size:
                                break
                    
                    if not chunk:
                        break
                    
                    total_bytes += len(chunk)
                    
                    part = await client.upload_part(
                        Bucket=self.bucket,
                        Key=object_name,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    parts.append({"PartNumber": part_number, "ETag": part["ETag"]})
                    part_number += 1
                
                await client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=object_name,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
                return total_bytes
            except Exception as e:
                logger.exception("Multipart upload failed for %s. Aborting...", object_name)
                try:
                    await client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=object_name,
                        UploadId=upload_id,
                    )
                except Exception:
                    logger.exception("Failed to abort multipart upload for %s", object_name)
                raise e

    async def download_bytes(
        self,
        object_name: str,
    ) -> bytes:
        async with self.session.client(**self._get_client_args()) as client:
            response = await client.get_object(
                Bucket=self.bucket,
                Key=object_name,
            )
            async with response["Body"] as stream:
                return await stream.read()

    async def delete(
        self,
        object_name: str,
    ) -> None:
        async with self.session.client(**self._get_client_args()) as client:
            await client.delete_object(
                Bucket=self.bucket,
                Key=object_name,
            )

