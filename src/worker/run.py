import asyncio
import logging

from src.database import db
from src.queue import queue
from src.worker.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    worker = Worker()

    await db.connect()
    await queue.connect()

    logger.info("Worker standalone mode started")
    try:
        await worker.start()
    except asyncio.CancelledError:
        pass
    finally:
        await worker.stop()
        await queue.close()
        await db.close()
        logger.info("Worker standalone mode stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
