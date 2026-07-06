import asyncio
import json
import logging

from src.configs import settings
from src.core.pipeline import Pipeline
from src.database import db
from src.database.models import JobModel
from src.models.job import IngestionJob, JobStatus
from src.queue import queue

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self) -> None:
        self.pipeline = Pipeline()
        self._running = False

    async def _process_job(self, raw_message: bytes) -> None:
        try:
            job_data = json.loads(raw_message)
            job = IngestionJob(**job_data)

            logger.info("Processing job %s with %d files", job.job_id, len(job.files))

            await self._update_job_status(job.job_id, JobStatus.processing)

            result = await self.pipeline.run(
                job_id=job.job_id,
                files=[f.model_dump() for f in job.files],
            )

            if result.errors:
                logger.warning(
                    "Job %s completed with errors: %s",
                    job.job_id,
                    result.errors,
                )

            await self._update_job_status(job.job_id, JobStatus.completed)

            logger.info(
                "Job %s completed: %d chunks embedded",
                job.job_id,
                result.total_chunks,
            )

        except Exception as e:
            logger.exception("Job processing failed: %s", e)
            try:
                job_data = json.loads(raw_message)
                await self._update_job_status(job_data["job_id"], JobStatus.failed)
            except Exception:
                logger.exception("Failed to update job status to failed")

    async def _update_job_status(self, job_id: str, status: JobStatus) -> None:
        async with db.pool() as session:
            job = await session.get(JobModel, job_id)
            if job:
                job.status = status.value
                await session.commit()

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
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self._running = False
        logger.info("Worker stopped")
