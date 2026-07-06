from fastapi import APIRouter

from src.configs import settings

from src.api.v1 import models, upload


api_router = APIRouter(prefix=settings.api_prefix)

api_router.include_router(upload.router)
api_router.include_router(models.router)
