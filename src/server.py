from fastapi import FastAPI
import uvicorn

from src.configs import settings

app = FastAPI(
    title=settings.app.app_name,
    version=settings.app.app_version,
)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


def main() -> None:
    uvicorn.run(
        app,
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
    )


if __name__ == "__main__":
    main()