from fastapi import FastAPI
import uvicorn

from src.configs.app import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


def main() -> None:
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()