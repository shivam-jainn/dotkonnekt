from uuid import uuid4
import io
import uuid
import pytest

from src.storage.s3 import S3Storage


@pytest.mark.integration
class TestMinioStorageIntegration:
    @pytest.fixture(autouse=True)
    def setup_storage(self):
        self.storage = S3Storage()
        # We don't have a synchronous way to clear bucket easily here,
        # but objects are random UUIDs so they shouldn't conflict.

    @pytest.mark.asyncio
    async def test_upload_and_download_bytes(self):
        storage = self.storage
        object_name = f"test-file-{uuid.uuid4()}.txt"
        content = b"hello integration test"

        await storage.upload_bytes(content, object_name, "text/plain")

        retrieved = await storage.download_bytes(object_name)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_upload_and_delete(self):
        storage = self.storage
        object_name = f"test-delete-{uuid.uuid4()}.txt"

        await storage.upload_bytes(b"temporary", object_name, "text/plain")
        
        await storage.delete(object_name)

        with pytest.raises(Exception):
            await storage.download_bytes(object_name)

    @pytest.mark.asyncio
    async def test_upload_overwrite(self):
        storage = self.storage
        object_name = f"test-overwrite-{uuid.uuid4()}.txt"

        await storage.upload_bytes(b"version-1", object_name, "text/plain")
        await storage.upload_bytes(b"version-2", object_name, "text/plain")

        retrieved = await storage.download_bytes(object_name)
        assert retrieved == b"version-2"

    @pytest.mark.asyncio
    async def test_upload_stream(self):
        storage = self.storage
        object_name = f"test-stream-{uuid.uuid4()}.bin"
        
        data = b"a" * (6 * 1024 * 1024)
        stream = io.BytesIO(data)

        await storage.upload_stream(stream, object_name, "application/octet-stream")

        retrieved = await storage.download_bytes(object_name)
        assert len(retrieved) == len(data)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_multiple_files_in_prefix(self):
        storage = self.storage
        prefix = f"job-{uuid.uuid4()}/"
        
        files = {
            f"{prefix}file1.txt": b"content1",
            f"{prefix}file2.txt": b"content2",
        }

        for obj_name, content in files.items():
            await storage.upload_bytes(content, obj_name, "text/plain")

        for obj_name, expected_content in files.items():
            retrieved = await storage.download_bytes(obj_name)
            assert retrieved == expected_content

    @pytest.mark.asyncio
    async def test_download_non_existent_raises(self):
        storage = self.storage
        object_name = f"does-not-exist-{uuid.uuid4()}.txt"

        with pytest.raises(Exception):
            await storage.download_bytes(object_name)
