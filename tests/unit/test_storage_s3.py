from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from src.storage.s3 import S3Storage


@pytest.mark.unit
class TestS3Storage:
    @patch("src.storage.s3.aioboto3.Session")
    async def test_upload_bytes_calls_put_object(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm

        storage = S3Storage()
        await storage.upload_bytes(b"file content", "path/to/file.txt", "text/plain")

        client.put_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
            Body=b"file content",
            ContentType="text/plain",
        )

    @patch("src.storage.s3.aioboto3.Session")
    async def test_download_bytes_calls_get_object_and_returns_body(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm
        
        body_stream = AsyncMock()
        body_stream.read.return_value = b"downloaded"
        body_stream_cm = AsyncMock()
        body_stream_cm.__aenter__.return_value = body_stream
        
        client.get_object.return_value = {"Body": body_stream_cm}

        storage = S3Storage()
        result = await storage.download_bytes("path/to/file.txt")

        client.get_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
        )
        assert result == b"downloaded"

    @patch("src.storage.s3.aioboto3.Session")
    async def test_delete_calls_delete_object(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm

        storage = S3Storage()
        await storage.delete("path/to/file.txt")

        client.delete_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
        )

    @patch("src.storage.s3.aioboto3.Session")
    async def test_upload_stream_calls_multipart_upload(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm

        client.create_multipart_upload.return_value = {"UploadId": "upload-id-123"}
        client.upload_part.return_value = {"ETag": "etag-1"}

        stream = AsyncMock()
        stream.read.side_effect = [b"stream data", b""]
        
        storage = S3Storage()
        total_bytes = await storage.upload_stream(stream, "path/to/stream.bin", "application/octet-stream")

        client.create_multipart_upload.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/stream.bin",
            ContentType="application/octet-stream",
        )
        client.upload_part.assert_called_once()
        client.complete_multipart_upload.assert_called_once()
        assert total_bytes == len(b"stream data")

    @patch("src.storage.s3.aioboto3.Session")
    async def test_ensure_bucket_skips_create_if_exists(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm

        storage = S3Storage()
        await storage.upload_bytes(b"data", "test", "text/plain")

        client.head_bucket.assert_called_once_with(Bucket="dotkonnekt")
        client.create_bucket.assert_not_called()

    @patch("src.storage.s3.aioboto3.Session")
    async def test_ensure_bucket_creates_if_not_exists(self, mock_session_class):
        session = MagicMock()
        client = AsyncMock()
        mock_session_class.return_value = session
        
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = client
        session.client.return_value = client_cm
        client.head_bucket.side_effect = Exception("NotFound")

        storage = S3Storage()
        await storage.upload_bytes(b"data", "test", "text/plain")

        client.head_bucket.assert_called_once_with(Bucket="dotkonnekt")
        client.create_bucket.assert_called_once_with(Bucket="dotkonnekt")
