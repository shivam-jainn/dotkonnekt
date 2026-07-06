import json
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.configs import settings
from src.database import db
from src.database.models import JobModel
from src.models.job import FileInfo, IngestionJob
from src.queue import queue
from src.storage import create_storage

router = APIRouter(prefix="/documents", tags=["Documents"])


class UploadResponse(BaseModel):
    job_id: str
    status: str
    files_uploaded: int


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload documents for ingestion",
)
async def upload_documents(
    files: Annotated[list[UploadFile], File(...)],
    collection: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    job_id = str(uuid4())
    parsed_metadata = json.loads(metadata) if metadata else {}
    uploaded_files: list[FileInfo] = []
    storage = create_storage()

    for file in files:
        contents = await file.read()
        object_name = f"{job_id}/{file.filename}"
        storage.upload_bytes(contents, object_name, file.content_type or "application/octet-stream")
        uploaded_files.append(
            FileInfo(
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                size=len(contents),
                storage_path=object_name,
            )
        )

    job = IngestionJob(
        job_id=job_id,
        files=uploaded_files,
        collection=collection,
        metadata=parsed_metadata,
    )

    async with db.pool() as session:
        session.add(JobModel(
            id=job.job_id,
            status=job.status.value,
            files=[f.model_dump() for f in job.files],
            collection=job.collection,
            metadata_=job.metadata,
        ))
        await session.commit()

    await queue.publish(
        settings.rabbitmq_queue,
        job.model_dump_json().encode(),
    )

    return UploadResponse(
        job_id=job_id,
        status="queued",
        files_uploaded=len(uploaded_files),
    )
