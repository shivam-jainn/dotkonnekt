from fastapi import APIRouter

from src.configs import settings

from src.api.v1 import upload


api_router = APIRouter(prefix=settings.app.api_prefix)

api_router.include_router(upload.router)