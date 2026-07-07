import asyncio
import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.configs import settings
from src.core.document import Chunk as DocChunk
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
                try:
                    await self._client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        ),
                    )
                except UnexpectedResponse as e:
                    if "already exists" in str(e):
                        logger.info(
                            "Collection '%s' already exists (race), using it",
                            collection,
                        )
                    else:
                        raise

            self._collection_cache.add(collection)

    async def recreate_collection(self, collection: str, vector_size: int) -> None:
        """Force delete and recreate a collection with a new vector dimension size."""
        async with self._ensure_lock:
            logger.warning(
                "Deleting and recreating collection '%s' due to schema/dimension mismatch (new size=%d)",
                collection,
                vector_size,
            )
            try:
                await self._client.delete_collection(collection_name=collection)
            except Exception:
                pass
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            self._collection_cache.add(collection)

    async def initialize(self) -> None:
        """Pre-create the default Qdrant collection at startup.

        Uses a safe default vector size since no embedding has been computed yet;
        the size is updated on the first ``store_batch`` or ``store_document_chunks``
        call via ``ensure_collection``.
        """
        logger.info("Initialising Qdrant collection '%s'", settings.qdrant_collection)

        default_size = settings.qdrant_vector_size

        exists = await self._client.collection_exists(settings.qdrant_collection)
        if exists:
            logger.info(
                "Qdrant collection '%s' already exists, skipping creation",
                settings.qdrant_collection,
            )
            self._collection_cache.add(settings.qdrant_collection)
            return

        logger.info(
            "Creating Qdrant collection '%s' (size=%d)",
            settings.qdrant_collection,
            default_size,
        )
        try:
            await self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=default_size, distance=Distance.COSINE
                ),
            )
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.info(
                    "Collection '%s' already exists (race), using it",
                    settings.qdrant_collection,
                )
            else:
                raise

        self._collection_cache.add(settings.qdrant_collection)
        logger.info(
            "Qdrant collection '%s' initialised", settings.qdrant_collection
        )

    async def store_batch(self, collection: str, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        vector_size = len(chunks[0].embedding)
        await self.ensure_collection(collection, vector_size)

        points = [
            PointStruct(
                id=chunk.id if chunk.id else str(
                    uuid.uuid5(
                        QDRANT_NS,
                        f"{chunk.metadata.get('job_id', 'doc')}_{chunk.metadata.get('filename', 'doc')}_{chunk.index}",
                    )
                ),
                vector=chunk.embedding,
                payload=chunk.to_payload(),
            )
            for chunk in chunks
        ]

        try:
            await self._client.upsert(
                collection_name=collection,
                points=points,
            )
        except UnexpectedResponse as e:
            e_str = str(e).lower()
            content_str = ""
            if hasattr(e, "content") and e.content:
                content_str = e.content.decode("utf-8", errors="ignore").lower()

            if "not found" in e_str or "doesn't exist" in e_str or "not found" in content_str or "doesn't exist" in content_str:
                logger.warning(
                    "Collection '%s' not found during upsert despite cache. Invaliding cache and recreating...",
                    collection,
                )
                self._collection_cache.discard(collection)
                await self.ensure_collection(collection, vector_size)
                await self._client.upsert(
                    collection_name=collection,
                    points=points,
                )
            elif "dimension" in e_str or "expected dim" in e_str or "dimension" in content_str or "expected dim" in content_str:
                logger.warning(
                    "Collection '%s' has a vector dimension mismatch. Deleting and recreating...",
                    collection,
                )
                self._collection_cache.discard(collection)
                await self.recreate_collection(collection, vector_size)
                await self._client.upsert(
                    collection_name=collection,
                    points=points,
                )
            else:
                raise

        logger.debug(
            "Stored %d points to Qdrant collection '%s'", len(points), collection
        )

    async def store_document_chunks(
        self, collection: str, chunks: list[DocChunk], embeddings: list[list[float]]
    ) -> None:
        """Store Document IR chunks directly with pre-computed embeddings."""
        if not chunks or not embeddings:
            return

        vector_size = len(embeddings[0])
        await self.ensure_collection(collection, vector_size)

        points = [
            PointStruct(
                id=chunk.id if chunk.id else str(
                    uuid.uuid5(
                        QDRANT_NS,
                        f"{chunk.metadata.get('job_id', 'doc')}_{chunk.metadata.get('filename', 'doc')}_{chunk.index}",
                    )
                ),
                vector=emb,
                payload=chunk.to_payload(),
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        try:
            await self._client.upsert(
                collection_name=collection,
                points=points,
            )
        except UnexpectedResponse as e:
            e_str = str(e).lower()
            content_str = ""
            if hasattr(e, "content") and e.content:
                content_str = e.content.decode("utf-8", errors="ignore").lower()

            if "not found" in e_str or "doesn't exist" in e_str or "not found" in content_str or "doesn't exist" in content_str:
                logger.warning(
                    "Collection '%s' not found during upsert despite cache. Invaliding cache and recreating...",
                    collection,
                )
                self._collection_cache.discard(collection)
                await self.ensure_collection(collection, vector_size)
                await self._client.upsert(
                    collection_name=collection,
                    points=points,
                )
            elif "dimension" in e_str or "expected dim" in e_str or "dimension" in content_str or "expected dim" in content_str:
                logger.warning(
                    "Collection '%s' has a vector dimension mismatch. Deleting and recreating...",
                    collection,
                )
                self._collection_cache.discard(collection)
                await self.recreate_collection(collection, vector_size)
                await self._client.upsert(
                    collection_name=collection,
                    points=points,
                )
            else:
                raise

        logger.debug(
            "Stored %d document chunks to Qdrant collection '%s'",
            len(points),
            collection,
        )

    async def close(self) -> None:
        await self._client.close()
