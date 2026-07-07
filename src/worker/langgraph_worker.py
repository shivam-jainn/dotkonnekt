import asyncio
import json
import logging
import random

from src.configs import settings
from src.database import db
from src.database.models import JobModel
from src.models.job import JobStatus
from src.models.langgraph_message import LangGraphMessage
from src.queue import queue

logger = logging.getLogger(__name__)


class LangGraphWorker:
    """
    Worker that consumes LangGraphMessage jobs from the queue, runs the LangGraph graph, and persists the reconciled report to the database.
    """

    def __init__(self) -> None:
        self._running = False
        self._stop_event = asyncio.Event()

    async def _invoke_with_retry(self, app, initial_state: dict, config: dict) -> dict:
        max_retries = settings.langgraph_max_retries
        base_delay = settings.langgraph_retry_base_delay
        max_delay = settings.langgraph_retry_max_delay

        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await app.ainvoke(initial_state, config=config)
            except Exception as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    break  # exhausted — bubble up below

                delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
                logger.warning(
                    "[LangGraphWorker] ainvoke failed (attempt %d/%d) for job %s. "
                    "Retrying in %.1fs — %s",
                    attempt + 1,
                    max_retries,
                    initial_state.get("job_id"),
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"LangGraph invocation failed after {max_retries} attempts"
        ) from last_exc

    async def _process_job(self, raw_message: bytes) -> None:
        """Parse a LangGraphMessage, run the graph, persist the report."""
        msg = LangGraphMessage.model_validate_json(raw_message)
        job_id = msg.job_id

        logger.info(
            "[LangGraphWorker] Received job %s with %d chunks",
            job_id,
            len(msg.chunks),
        )

        try:
            from src.core.agent.graph import create_graph
            from src.core.tracing.collector import StructuredTraceCollector

            app = create_graph()
            tracer = StructuredTraceCollector(job_id=job_id)

            initial_state = {
                "job_id": job_id,
                # Pass chunks as plain dicts so graph nodes can access them via
                # state["chunks"]; graph_batch_size controls batching inside the
                # process_chunk_node loop.
                "chunks": [c.model_dump() for c in msg.chunks],
                "files": [],          # already ingested upstream
                "extracted_images": [],
                "current_chunk_idx": 0,
                "findings": [],
                "errors": [],
                "overall_score": 0.0,
                "reconciled_report": None,
            }

            config = {
                "callbacks": [tracer],
                "configurable": {"thread_id": str(job_id)},
            }

            logger.info(
                "[LangGraphWorker] Invoking graph for job %s (batch_size=%d)",
                job_id,
                settings.graph_batch_size,
            )

            # Run with exponential backoff
            result = await self._invoke_with_retry(app, initial_state, config)

            # Flush traces to DB
            await tracer.flush_to_db()

            if result.get("errors"):
                logger.warning(
                    "[LangGraphWorker] Job %s finished with errors: %s",
                    job_id,
                    result["errors"],
                )

            # Persist the reconciled report
            report = result.get("reconciled_report") or {}
            await self._save_report(job_id, report)

            await self._update_job_status(job_id, JobStatus.completed)

            logger.info(
                "[LangGraphWorker] Job %s analysis complete — score: %s",
                job_id,
                report.get("score"),
            )

        except Exception as exc:
            logger.exception(
                "[LangGraphWorker] Job %s failed after all retries: %s", job_id, exc
            )
            await self._update_job_status(job_id, JobStatus.failed)
            raise

    async def _save_report(self, job_id: str, report: dict) -> None:
        """Persist the reconciled LangGraph report into the jobs table."""
        async with db.pool() as session:
            from sqlalchemy import text

            await session.execute(
                text(
                    "UPDATE jobs "
                    "SET metadata = jsonb_set(metadata, '{report}', cast(:report as jsonb)) "
                    "WHERE id = :job_id"
                ),
                {"report": json.dumps(report), "job_id": job_id},
            )
            await session.commit()
        logger.debug("[LangGraphWorker] Saved report for job %s", job_id)

    async def _update_job_status(self, job_id: str, status: JobStatus) -> None:
        async with db.pool() as session:
            job = await session.get(JobModel, job_id)
            if job:
                job.status = status.value
                await session.commit()

    async def start(self) -> None:
        if self._running:
            logger.warning("[LangGraphWorker] Already running")
            return

        self._running = True
        logger.info(
            "[LangGraphWorker] Starting, consuming from queue: %s",
            settings.langgraph_queue,
        )

        await queue.consume(settings.langgraph_queue, self._process_job)

        logger.info("[LangGraphWorker] Now listening for messages")
        self._stop_event.clear()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        logger.info("[LangGraphWorker] Stopped")
