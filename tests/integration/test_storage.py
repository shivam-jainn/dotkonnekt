from uuid import uuid4

import pytest

from src.storage.minio import MinioStorage


pytestmark = pytest.mark.integration


@pytest.fixture
def storage():
    s = MinioStorage()
    yield s
    # Cleanup all test objects
    objects = list(s.client.list_objects(s.bucket, prefix="test-", recursive=True))
    for obj in objects:
        s.delete(obj.object_name)


class TestMinioStorageIntegration:
    def test_upload_and_download_bytes(self, storage):
        object_name = f"test-{uuid4().hex}/hello.txt"
        content = b"Hello, MinIO!"

        storage.upload_bytes(content, object_name, "text/plain")
        retrieved = storage.download_bytes(object_name)

        assert retrieved == content

    def test_upload_and_delete(self, storage):
        object_name = f"test-{uuid4().hex}/temp.txt"
        storage.upload_bytes(b"temporary", object_name, "text/plain")

        storage.delete(object_name)

        with pytest.raises(Exception):
            storage.download_bytes(object_name)

    def test_upload_overwrite(self, storage):
        object_name = f"test-{uuid4().hex}/overwrite.txt"

        storage.upload_bytes(b"version-1", object_name, "text/plain")
        storage.upload_bytes(b"version-2", object_name, "text/plain")

        retrieved = storage.download_bytes(object_name)
        assert retrieved == b"version-2"

    def test_upload_stream(self, storage):
        import io

        object_name = f"test-{uuid4().hex}/stream.bin"
        content = b"stream content here"
        stream = io.BytesIO(content)

        storage.upload_stream(stream, object_name, "application/octet-stream")
        retrieved = storage.download_bytes(object_name)

        assert retrieved == content

    def test_multiple_files_in_prefix(self, storage):
        prefix = f"test-{uuid4().hex}"
        files = [
            (f"{prefix}/a.txt", b"aaa"),
            (f"{prefix}/b.txt", b"bbb"),
            (f"{prefix}/sub/c.txt", b"ccc"),
        ]

        for obj_name, content in files:
            storage.upload_bytes(content, obj_name, "text/plain")

        objects = list(
            storage.client.list_objects(storage.bucket, prefix=prefix, recursive=True)
        )
        object_names = {o.object_name for o in objects}

        for obj_name, _ in files:
            assert obj_name in object_names

    def test_download_non_existent_raises(self, storage):
        object_name = f"test-{uuid4().hex}/nonexistent.txt"

        with pytest.raises(Exception):
            storage.download_bytes(object_name)
