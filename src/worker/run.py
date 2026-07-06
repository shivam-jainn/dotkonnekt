import argparse
import asyncio
import logging

from src.database import db
from src.queue import queue
from src.worker.storage_worker import StorageWorker
from src.worker.worker import Worker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def run_ingestion_worker() -> None:
    worker = Worker()

    await db.connect()
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
    storage_worker = StorageWorker()

    await queue.connect()

    logger.info("Storage worker started")
    try:
        await storage_worker.start()
    except asyncio.CancelledError:
        pass
    finally:
        await storage_worker.stop()
        await queue.close()
        logger.info("Storage worker stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="dotkonnekt worker")
    parser.add_argument(
        "--worker",
        choices=["ingestion", "storage"],
        default="ingestion",
        help="Worker type to run (default: ingestion)",
    )
    args = parser.parse_args()

    if args.worker == "ingestion":
        asyncio.run(run_ingestion_worker())
    elif args.worker == "storage":
        asyncio.run(run_storage_worker())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
