import asyncio
import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.configs import settings
from src.core.embedders.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

QDRANT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "qdrant.dotkonnekt.local")


class VectorStorer:
    def __init__(self) -> None:
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False,
            check_compatibility=False,
        )
        self._collection_cache: set[str] = set()
        self._ensure_lock = asyncio.Lock()

    async def ensure_collection(self, collection: str, vector_size: int) -> None:
        if collection in self._collection_cache:
            return

        async with self._ensure_lock:
            if collection in self._collection_cache:
                return

            exists = await self._client.collection_exists(collection)
            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' (size=%d)",
                    collection,
                    vector_size,
                )
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )

            self._collection_cache.add(collection)

    async def store_batch(self, collection: str, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        vector_size = len(chunks[0].embedding)
        await self.ensure_collection(collection, vector_size)

        points = [
            PointStruct(
                id=uuid.uuid5(
                    QDRANT_NS,
                    f"{chunk.metadata.get('job_id', 'doc')}_{chunk.metadata.get('filename', 'doc')}_{chunk.index}",
                ),
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "index": chunk.index,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        await self._client.upsert(
            collection_name=collection,
            points=points,
        )

        logger.debug(
            "Stored %d points to Qdrant collection '%s'", len(points), collection
        )

    async def close(self) -> None:
        await self._client.close()
