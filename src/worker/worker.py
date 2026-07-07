import asyncio
import json
import logging

from src.configs import settings
from src.core.pipeline import Pipeline
from src.database import db
from src.database.models import JobModel
from src.models.job import IngestionJob, JobStatus
from src.models.langgraph_message import LangGraphMessage, RawChunk
from src.models.storage import StorageMessage, StoredChunk
from src.queue import queue

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self) -> None:
        self.pipeline = Pipeline()
        self._running = False
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    #  Downstream publishers                                               #
    # ------------------------------------------------------------------ #

    async def _publish_for_storage(self, job: IngestionJob, result) -> None:
        """Publish embedded chunks in batches to the storage queue."""
        if not result.embedded_chunks:
            return

        collection = job.collection or settings.qdrant_collection
        batch_size = settings.storage_batch_size

        for i in range(0, len(result.embedded_chunks), batch_size):
            batch = result.embedded_chunks[i : i + batch_size]
            message = StorageMessage(
                job_id=job.job_id,
                collection=collection,
                chunks=[
                    StoredChunk(
                        content=ec.content,
                        embedding=ec.embedding,
                        index=ec.index,
                        metadata={**ec.metadata, "job_id": job.job_id},
                    )
                    for ec in batch
                ],
            )
            await queue.publish(
                settings.storage_queue,
                message.model_dump_json().encode(),
            )

        logger.info(
            "Published %d chunks to storage queue for job %s",
            len(result.embedded_chunks),
            job.job_id,
        )

    async def _publish_for_langgraph(self, job: IngestionJob, result) -> None:
        """Publish raw (un-analysed) chunks to the langgraph queue.

        The LangGraphWorker will consume these and run the LLM agent graph
        to produce structured findings (obligations, entities, risky terms).
        """
        if not result.embedded_chunks:
            return

        collection = job.collection or settings.qdrant_collection
        message = LangGraphMessage(
            job_id=job.job_id,
            collection=collection,
            chunks=[
                RawChunk(
                    content=ec.content,
                    index=ec.index,
                    metadata=ec.metadata,
                )
                for ec in result.embedded_chunks
            ],
        )
        await queue.publish(
            settings.langgraph_queue,
            message.model_dump_json().encode(),
        )

        logger.info(
            "Published %d raw chunks to langgraph queue for job %s",
            len(result.embedded_chunks),
            job.job_id,
        )

    # ------------------------------------------------------------------ #
    #  Job processing                                                      #
    # ------------------------------------------------------------------ #

    async def _process_job(self, raw_message: bytes) -> None:
        try:
            job_data = json.loads(raw_message)
            job = IngestionJob(**job_data)

            logger.info("Processing job %s with %d files", job.job_id, len(job.files))

            await self._update_job_status(job.job_id, JobStatus.processing)

            # Run the parse → chunk → embed pipeline
            files_dict = [f.model_dump() for f in job.files]
            result = await self.pipeline.run(job.job_id, files_dict)

            # Fan out to both downstream queues:
            #   1. Storage queue   — embeddings for Qdrant upsert
            #   2. LangGraph queue — raw chunks for LLM analysis
            await self._publish_for_storage(job, result)
            await self._publish_for_langgraph(job, result)

            # Status stays "processing" — the LangGraphWorker will flip it to
            # "completed" or "failed" once analysis finishes.
            logger.info(
                "Job %s ingestion done: %d chunks → storage + langgraph queues",
                job.job_id,
                len(result.embedded_chunks) if result.embedded_chunks else 0,
            )

        except Exception as e:
            logger.exception("Job processing failed: %s", e)
            try:
                job_data = json.loads(raw_message)
                await self._update_job_status(job_data["job_id"], JobStatus.failed)
            except Exception:
                logger.exception("Failed to update job status to failed")
            raise e

    # ------------------------------------------------------------------ #
    #  DB helpers                                                          #
    # ------------------------------------------------------------------ #

    async def _update_job_status(self, job_id: str, status: JobStatus) -> None:
        async with db.pool() as session:
            job = await session.get(JobModel, job_id)
            if job:
                job.status = status.value
                await session.commit()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        logger.info(
            "Starting worker, consuming from queue: %s", settings.rabbitmq_queue
        )

        await queue.consume(settings.rabbitmq_queue, self._process_job)

        logger.info("Worker is now listening for messages")
        self._stop_event.clear()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        logger.info("Worker stopped")
