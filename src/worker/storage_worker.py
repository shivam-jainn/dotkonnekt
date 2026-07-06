import asyncio
import logging

from src.configs import settings
from src.core.embedders.embedder import EmbeddedChunk
from src.core.storer import VectorStorer
from src.models.storage import StorageMessage
from src.queue import queue

logger = logging.getLogger(__name__)


class StorageWorker:
    def __init__(self) -> None:
        self.storer = VectorStorer()
        self._running = False

    async def _store_chunks(self, raw_message: bytes) -> None:
        try:
            msg = StorageMessage.model_validate_json(raw_message)

            chunks = [
                EmbeddedChunk(
                    content=c.content,
                    embedding=c.embedding,
                    index=c.index,
                    metadata=c.metadata,
                )
                for c in msg.chunks
            ]

            await self.storer.store_batch(msg.collection, chunks)

            logger.info(
                "Stored %d chunks from job %s to collection '%s'",
                len(chunks),
                msg.job_id,
                msg.collection,
            )

        except Exception:
            logger.exception("Failed to store chunk batch — message will be requeued")
            raise

    async def start(self) -> None:
        if self._running:
            logger.warning("Storage worker is already running")
            return

        self._running = True
        logger.info(
            "Starting storage worker, consuming from queue: %s",
            settings.storage_queue,
        )

        await queue.consume(settings.storage_queue, self._store_chunks)

        logger.info("Storage worker is now listening for messages")
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self._running = False
        await self.storer.close()
        logger.info("Storage worker stopped")
