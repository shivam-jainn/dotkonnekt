from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.router import api_router
from src.configs import settings
from src.database import db
from src.queue import queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    await queue.connect()
    yield
    await queue.close()
    await db.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health():
    db_healthy = await db.health_check()
    if not db_healthy:
        return {"status": "degraded", "database": "unreachable"}
    return {"status": "healthy", "database": "connected"}


def main() -> None:
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()