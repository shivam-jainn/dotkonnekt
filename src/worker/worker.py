import asyncio
import json
import logging

from src.configs import settings
from src.core.pipeline import Pipeline
from src.database import db
from src.database.models import JobModel
from src.models.job import IngestionJob, JobStatus
from src.models.langgraph_message import LangGraphMessage, AnalysisChunkModel
from src.models.storage import StorageMessage, StoredChunk
from src.queue import queue

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, enable_semantic_enrichment: bool = False) -> None:
        self.pipeline = Pipeline(
            enable_semantic_enrichment=enable_semantic_enrichment,
        )
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
                        id=ec.id,
                        content=ec.content,
                        embedding=ec.embedding,
                        index=ec.index,
                        metadata={**ec.metadata, "job_id": job.job_id},
                        page=ec.page,
                        section=ec.section,
                        subsection=ec.subsection,
                        clause=ec.clause,
                        previous_chunk=ec.previous_chunk_id,
                        next_chunk=ec.next_chunk_id,
                        summary=ec.semantic_metadata.get("summary") if ec.semantic_metadata else None,
                        keywords=ec.semantic_metadata.get("keywords", []) if ec.semantic_metadata else [],
                        obligations=ec.semantic_metadata.get("obligations", []) if ec.semantic_metadata else [],
                        entities=ec.semantic_metadata.get("entities", []) if ec.semantic_metadata else [],
                        risks=ec.semantic_metadata.get("risks", []) if ec.semantic_metadata else [],
                        document_type=ec.metadata.get("document_type"),
                        party_sentences=ec.semantic_metadata.get("party_sentences", {}) if ec.semantic_metadata else {},
                        obligations_by_party=ec.semantic_metadata.get("obligations_by_party", []) if ec.semantic_metadata else [],
                        risks_by_party=ec.semantic_metadata.get("risks_by_party", []) if ec.semantic_metadata else [],
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
        """Publish raw (un-analysed) chunks to the langgraph queue."""
        if not result.embedded_chunks:
            return

        collection = job.collection or settings.qdrant_collection
        message = LangGraphMessage(
            job_id=job.job_id,
            collection=collection,
            chunks=[
                AnalysisChunkModel(
                    id=ec.id,
                    content=ec.content,
                    index=ec.index,
                    page=ec.page,
                    section=ec.section,
                    subsection=ec.subsection,
                    clause=ec.clause,
                    previous_chunk=ec.previous_chunk_id,
                    next_chunk=ec.next_chunk_id,
                    clause_id=ec.clause_id,
                    derived_metadata=ec.semantic_metadata, # this has derived metadata format we created in Pipeline
                    content_type=ec.content_type,
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
            "Published %d chunks to langgraph queue for job %s",
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

            # Run the parse → chunk → enrich → embed pipeline
            files_dict = [f.model_dump() for f in job.files]
            result = await self.pipeline.run(job.job_id, files_dict)

            # Fan out to both downstream queues:
            #   1. Storage queue   — embeddings for Qdrant upsert
            #   2. LangGraph queue — raw chunks for LLM analysis
            await self._publish_for_storage(job, result)
            await self._publish_for_langgraph(job, result)

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
