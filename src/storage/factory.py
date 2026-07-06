from typing import Type

from src.configs import settings
from src.storage.base import Storage
from src.storage.minio import MinioStorage
from src.storage.s3 import S3Storage

_STORAGE_PROVIDERS: dict[str, Type[Storage]] = {
    "minio": MinioStorage,
    "s3": S3Storage,
}


def create_storage() -> Storage:
    try:
        storage_cls = _STORAGE_PROVIDERS[settings.storage.provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown storage provider '{settings.storage.provider}'. "
            f"Available: {', '.join(_STORAGE_PROVIDERS)}"
        ) from e

    return storage_cls()