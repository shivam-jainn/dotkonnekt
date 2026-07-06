import io
from unittest.mock import MagicMock, patch

import pytest

from src.storage.minio import MinioStorage


@pytest.mark.unit
class TestMinioStorage:
    @patch("src.storage.minio.Minio")
    def test_upload_bytes_calls_put_object(self, mock_minio_class):
        client = MagicMock()
        mock_minio_class.return_value = client

        storage = MinioStorage()
        storage.upload_bytes(b"file content", "path/to/file.txt", "text/plain")

        client.put_object.assert_called_once()
        call_kwargs = client.put_object.call_args[1]
        assert call_kwargs["bucket_name"] == "dotkonnekt"
        assert call_kwargs["object_name"] == "path/to/file.txt"
        assert call_kwargs["length"] == 12
        assert call_kwargs["content_type"] == "text/plain"
        assert isinstance(call_kwargs["data"], io.BytesIO)

    @patch("src.storage.minio.Minio")
    def test_download_bytes_calls_get_object_and_returns_bytes(self, mock_minio_class):
        client = MagicMock()
        mock_minio_class.return_value = client
        response = MagicMock()
        response.read.return_value = b"downloaded content"
        client.get_object.return_value = response

        storage = MinioStorage()
        result = storage.download_bytes("path/to/file.txt")

        client.get_object.assert_called_once_with(
            bucket_name="dotkonnekt",
            object_name="path/to/file.txt",
        )
        response.read.assert_called_once()
        assert result == b"downloaded content"

    @patch("src.storage.minio.Minio")
    def test_delete_calls_remove_object(self, mock_minio_class):
        client = MagicMock()
        mock_minio_class.return_value = client

        storage = MinioStorage()
        storage.delete("path/to/file.txt")

        client.remove_object.assert_called_once_with(
            bucket_name="dotkonnekt",
            object_name="path/to/file.txt",
        )

    @patch("src.storage.minio.Minio")
    def test_upload_stream_calls_put_object_with_part_size(self, mock_minio_class):
        client = MagicMock()
        mock_minio_class.return_value = client

        stream = io.BytesIO(b"stream data")
        storage = MinioStorage()
        storage.upload_stream(stream, "path/to/stream.bin", "application/octet-stream")

        client.put_object.assert_called_once()
        call_kwargs = client.put_object.call_args[1]
        assert call_kwargs["bucket_name"] == "dotkonnekt"
        assert call_kwargs["object_name"] == "path/to/stream.bin"
        assert call_kwargs["data"] is stream
        assert call_kwargs["length"] == -1
        assert call_kwargs["part_size"] == 10 * 1024 * 1024

    @patch("src.storage.minio.Minio")
    def test_ensure_bucket_does_not_create_if_exists(self, mock_minio_class):
        client = MagicMock()
        client.bucket_exists.return_value = True
        mock_minio_class.return_value = client

        storage = MinioStorage()

        client.bucket_exists.assert_called_once_with("dotkonnekt")
        client.make_bucket.assert_not_called()

    @patch("src.storage.minio.Minio")
    def test_ensure_bucket_creates_if_not_exists(self, mock_minio_class):
        client = MagicMock()
        client.bucket_exists.return_value = False
        mock_minio_class.return_value = client

        storage = MinioStorage()

        client.bucket_exists.assert_called_once_with("dotkonnekt")
        client.make_bucket.assert_called_once_with("dotkonnekt")
