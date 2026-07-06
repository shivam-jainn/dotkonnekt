from unittest.mock import MagicMock, patch

import pytest

from src.storage.s3 import S3Storage


@pytest.mark.unit
class TestS3Storage:
    @patch("src.storage.s3.boto3.Session")
    def test_upload_bytes_calls_put_object(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client

        storage = S3Storage()
        storage.upload_bytes(b"file content", "path/to/file.txt", "text/plain")

        client.put_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
            Body=b"file content",
            ContentType="text/plain",
        )

    @patch("src.storage.s3.boto3.Session")
    def test_download_bytes_calls_get_object_and_returns_body(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client
        client.get_object.return_value = {"Body": MagicMock()}
        client.get_object.return_value["Body"].read.return_value = b"downloaded"

        storage = S3Storage()
        result = storage.download_bytes("path/to/file.txt")

        client.get_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
        )
        assert result == b"downloaded"

    @patch("src.storage.s3.boto3.Session")
    def test_delete_calls_delete_object(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client

        storage = S3Storage()
        storage.delete("path/to/file.txt")

        client.delete_object.assert_called_once_with(
            Bucket="dotkonnekt",
            Key="path/to/file.txt",
        )

    @patch("src.storage.s3.boto3.Session")
    def test_upload_stream_calls_upload_fileobj(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client

        stream = MagicMock()
        storage = S3Storage()
        storage.upload_stream(stream, "path/to/stream.bin", "application/octet-stream")

        client.upload_fileobj.assert_called_once_with(
            Fileobj=stream,
            Bucket="dotkonnekt",
            Key="path/to/stream.bin",
            ExtraArgs={"ContentType": "application/octet-stream"},
        )

    @patch("src.storage.s3.boto3.Session")
    def test_ensure_bucket_skips_create_if_exists(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client

        S3Storage()

        client.head_bucket.assert_called_once_with(Bucket="dotkonnekt")
        client.create_bucket.assert_not_called()

    @patch("src.storage.s3.boto3.Session")
    def test_ensure_bucket_creates_if_not_exists(self, mock_session_class):
        session = MagicMock()
        client = MagicMock()
        mock_session_class.return_value = session
        session.client.return_value = client
        client.head_bucket.side_effect = Exception("NotFound")

        S3Storage()

        client.head_bucket.assert_called_once_with(Bucket="dotkonnekt")
        client.create_bucket.assert_called_once_with(Bucket="dotkonnekt")

    @patch("src.storage.s3.boto3.Session")
    def test_client_created_with_path_addressing_and_s3v4(self, mock_session_class):
        session = MagicMock()
        mock_session_class.return_value = session

        S3Storage()

        session.client.assert_called_once()
        call_args = session.client.call_args
        assert call_args[0][0] == "s3"
        config = call_args[1]["config"]
        assert config.s3 == {"addressing_style": "path"}
        assert config.signature_version == "s3v4"
