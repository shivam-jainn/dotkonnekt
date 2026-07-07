import argparse
import asyncio
import logging

from src.database import db
from src.queue import queue
from src.worker.langgraph_worker import LangGraphWorker
from src.worker.storage_worker import StorageWorker
from src.worker.worker import Worker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Runner functions
# ────────────────────────────────────────────────────────────────────────────

async def run_ingestion_worker() -> None:
    """Parse/chunk/embed worker. Fans out to storage + langgraph queues."""
    worker = Worker()

    await db.connect()
    await db.create_all()
    await queue.connect()

    logger.info("Ingestion worker started")
    try:
        await worker.start()
    except asyncio.CancelledError:
        pass
    finally:
        await worker.stop()
        await queue.close()
        await db.close()
        logger.info("Ingestion worker stopped")


async def run_storage_worker() -> None:
    """Storage worker. Reads embeddings from the storage queue → Qdrant."""
    storage_worker = StorageWorker()

    await db.connect()
    await db.create_all()
    await queue.connect()

    logger.info("Storage worker started")
    try:
        await storage_worker.start()
    except asyncio.CancelledError:
        pass
    finally:
        await storage_worker.stop()
        await queue.close()
        await db.close()
        logger.info("Storage worker stopped")


async def run_langgraph_worker() -> None:
    """LangGraph worker. Reads raw chunks from the langgraph queue,
    runs the LLM agent graph with batching + exponential backoff retry,
    and persists the reconciled report to the database."""
    langgraph_worker = LangGraphWorker()

    await db.connect()
    await db.create_all()
    await queue.connect()

    logger.info("LangGraph worker started")
    try:
        await langgraph_worker.start()
    except asyncio.CancelledError:
        pass
    finally:
        await langgraph_worker.stop()
        await queue.close()
        await db.close()
        logger.info("LangGraph worker stopped")


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="dotkonnekt worker")
    parser.add_argument(
        "--worker",
        choices=["ingestion", "storage", "langgraph"],
        default="ingestion",
        help=(
            "Worker type to run (default: ingestion).\n"
            "  ingestion  — parse/chunk/embed → fans out to storage + langgraph queues\n"
            "  storage    — upsert embeddings into Qdrant\n"
            "  langgraph  — run LLM agent graph on raw chunks with retry"
        ),
    )
    args = parser.parse_args()

    runners = {
        "ingestion": run_ingestion_worker,
        "storage": run_storage_worker,
        "langgraph": run_langgraph_worker,
    }

    asyncio.run(runners[args.worker]())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
