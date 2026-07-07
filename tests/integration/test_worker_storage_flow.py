import asyncio
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from src.configs import settings
from src.core.embedders.embedder import EmbeddedChunk
from src.core.pipeline import Pipeline, PipelineResult
from src.core.storer import VectorStorer
from src.models.job import IngestionJob
from src.queue import queue as module_queue
from src.worker.storage_worker import StorageWorker
from src.worker.worker import Worker


pytestmark = pytest.mark.integration


@pytest.fixture
async def qdrant_client():
    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        https=False,
        check_compatibility=False,
    )
    yield client
    await client.close()


@pytest.fixture
def test_collection():
    name = f"test-{uuid4().hex[:12]}"
    yield name


@pytest.fixture
def test_storage_queue():
    name = f"test-storage-{uuid4().hex[:8]}"
    original = settings.storage_queue
    settings.storage_queue = name
    yield name
    settings.storage_queue = original


@pytest.fixture(autouse=True)
async def connected_queue():
    await module_queue.connect()
    yield
    await module_queue.close()


def _make_pdf(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestWorkerToStorageFlow:
    async def test_publish_and_store_direct(
        self, qdrant_client, test_collection, test_storage_queue
    ):
        chunks = [
            EmbeddedChunk(
                content="Hello from chunk one",
                embedding=[0.1 + i * 0.01 for i in range(768)],
                index=0,
                metadata={"filename": "test.pdf", "chunk_index": 0},
            ),
            EmbeddedChunk(
                content="Hello from chunk two",
                embedding=[0.2 + i * 0.01 for i in range(768)],
                index=1,
                metadata={"filename": "test.pdf", "chunk_index": 1},
            ),
        ]

        result = PipelineResult(job_id="test-job", embedded_chunks=chunks)
        job = IngestionJob(job_id="test-job", collection=test_collection)
        worker = Worker()

        storage_worker = StorageWorker()
        consume_task = asyncio.create_task(
            module_queue.consume(settings.storage_queue, storage_worker._store_chunks)
        )
        await asyncio.sleep(0.3)

        await worker._publish_for_storage(job, result)
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        exists = await qdrant_client.collection_exists(test_collection)
        assert exists, f"Collection '{test_collection}' should exist"

        points, _ = await qdrant_client.scroll(
            collection_name=test_collection,
            limit=10,
        )
        assert len(points) == 2

        contents = {p.payload["content"] for p in points}
        assert contents == {"Hello from chunk one", "Hello from chunk two"}

    async def test_duplicate_upsert_is_idempotent(
        self, qdrant_client, test_collection, test_storage_queue
    ):
        chunks = [
            EmbeddedChunk(
                content="Dedup content",
                embedding=[0.5] * 768,
                index=0,
                metadata={"job_id": "dedup-job", "filename": "dedup.pdf", "chunk_index": 0},
            ),
        ]

        result = PipelineResult(job_id="dedup-job", embedded_chunks=chunks)
        job = IngestionJob(job_id="dedup-job", collection=test_collection)

        storer = VectorStorer()
        await storer.store_batch(test_collection, chunks)

        worker = Worker()
        storage_worker = StorageWorker()
        consume_task = asyncio.create_task(
            module_queue.consume(settings.storage_queue, storage_worker._store_chunks)
        )
        await asyncio.sleep(0.3)

        await worker._publish_for_storage(job, result)
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        points, _ = await qdrant_client.scroll(
            collection_name=test_collection,
            limit=10,
        )
        assert len(points) == 1, "Duplicate upsert should not create extra points"

    async def test_pipeline_to_storage_end_to_end(
        self, qdrant_client, test_collection, test_storage_queue
    ):
        pdf_data = _make_pdf("This is a test document for end-to-end integration test.")
        mock_storage = AsyncMock()
        mock_storage.download_bytes.return_value = pdf_data

        pipeline = Pipeline()
        pipeline.storage = mock_storage

        from src.core.embedders.embedder import Embedder

        class FakeEmbedder(Embedder):
            async def embed_chunks(self, chunks):
                return [
                    EmbeddedChunk(
                        content=c.content,
                        embedding=[0.01 * c.index + 0.001 * i for i in range(768)],
                        index=c.index,
                        metadata=c.metadata,
                    )
                    for c in chunks
                ]

        pipeline.embedder = FakeEmbedder()

        result = await pipeline.run(
            job_id="e2e-test",
            files=[
                {
                    "filename": "test.pdf",
                    "storage_path": "e2e-test/test.pdf",
                }
            ],
        )

        assert result.total_chunks > 0
        assert result.errors == []

        job = IngestionJob(job_id="e2e-test", collection=test_collection)
        worker = Worker()

        storage_worker = StorageWorker()
        consume_task = asyncio.create_task(
            module_queue.consume(settings.storage_queue, storage_worker._store_chunks)
        )
        await asyncio.sleep(0.3)

        await worker._publish_for_storage(job, result)
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        exists = await qdrant_client.collection_exists(test_collection)
        assert exists, f"Collection '{test_collection}' should exist"

        points, _ = await qdrant_client.scroll(
            collection_name=test_collection,
            limit=100,
        )
        assert len(points) == result.total_chunks

        stored_contents = {p.payload["content"] for p in points}
        for chunk in result.embedded_chunks:
            assert chunk.content in stored_contents

    async def test_concurrent_storage_creates_collection_once(
        self, qdrant_client, test_collection, test_storage_queue
    ):
        chunks_batch_a = [
            EmbeddedChunk(
                content="Batch A content",
                embedding=[0.3] * 768,
                index=0,
                metadata={"filename": "a.pdf", "chunk_index": 0},
            ),
        ]
        chunks_batch_b = [
            EmbeddedChunk(
                content="Batch B content",
                embedding=[0.4] * 768,
                index=0,
                metadata={"filename": "b.pdf", "chunk_index": 0},
            ),
        ]

        result_a = PipelineResult(job_id="concurrent-a", embedded_chunks=chunks_batch_a)
        result_b = PipelineResult(job_id="concurrent-b", embedded_chunks=chunks_batch_b)
        job_a = IngestionJob(job_id="concurrent-a", collection=test_collection)
        job_b = IngestionJob(job_id="concurrent-b", collection=test_collection)

        worker = Worker()

        storage_worker = StorageWorker()
        consume_task = asyncio.create_task(
            module_queue.consume(settings.storage_queue, storage_worker._store_chunks)
        )
        await asyncio.sleep(0.3)

        await asyncio.gather(
            worker._publish_for_storage(job_a, result_a),
            worker._publish_for_storage(job_b, result_b),
        )
        await asyncio.sleep(0.5)

        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        points, _ = await qdrant_client.scroll(
            collection_name=test_collection,
            limit=10,
        )
        assert len(points) == 2

        exists = await qdrant_client.collection_exists(test_collection)
        assert exists, "Collection should have been created exactly once"
