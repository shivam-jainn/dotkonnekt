from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class FileInfo(BaseModel):
    filename: str
    content_type: str
    size: int
    storage_path: str


class IngestionJob(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    files: list[FileInfo] = Field(default_factory=list)
    collection: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
