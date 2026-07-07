import json
import logging
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
logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    job_id: str
    status: str
    files_uploaded: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    files: list[dict]
    collection: str | None
    metadata: dict
    created_at: str
    updated_at: str


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

    try:
        for file in files:
            object_name = f"{job_id}/{file.filename}"
            # Stream the upload and get the exact size
            size = await storage.upload_stream(
                file, object_name, file.content_type or "application/octet-stream"
            )
            uploaded_files.append(
                FileInfo(
                    filename=file.filename or "unknown",
                    content_type=file.content_type or "application/octet-stream",
                    size=size,
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
            session.add(
                JobModel(
                    id=job.job_id,
                    status=job.status.value,
                    files=[f.model_dump() for f in job.files],
                    collection=job.collection,
                    metadata_=job.metadata,
                )
            )
            
            try:
                # Atomic publish before committing transaction
                await queue.publish(
                    settings.rabbitmq_queue,
                    job.model_dump_json().encode(),
                )
                await session.commit()
            except Exception as publish_error:
                logger.error("Failed to publish ingestion job to queue: %s", publish_error)
                await session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to queue ingestion job. Transaction rolled back.",
                )

        return UploadResponse(
            job_id=job_id,
            status="queued",
            files_uploaded=len(uploaded_files),
        )

    except HTTPException:
        # Clean up uploaded files in case of failure/rollback
        for file_info in uploaded_files:
            try:
                await storage.delete(file_info.storage_path)
            except Exception:
                logger.exception("Failed to clean up uploaded file %s", file_info.storage_path)
        raise
    except Exception as e:
        logger.exception("In-flight upload failed")
        for file_info in uploaded_files:
            try:
                await storage.delete(file_info.storage_path)
            except Exception:
                logger.exception("Failed to clean up uploaded file %s", file_info.storage_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get status of an ingestion job",
)
async def get_job_status(job_id: str):
    async with db.pool() as session:
        job = await session.get(JobModel, job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )
        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            files=job.files,
            collection=job.collection,
            metadata=job.metadata_,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
        )


from sse_starlette.sse import EventSourceResponse
import asyncio

@router.get("/{job_id}/stream")
async def stream_job_progress(job_id: str):
    async def event_generator():
        last_trace_count = 0
        while True:
            async with db.pool() as session:
                job = await session.get(JobModel, job_id)
                if not job:
                    yield {"event": "error", "data": "Job not found"}
                    break
                    
                traces = job.metadata_.get("traces", []) if job.metadata_ else []
                if len(traces) > last_trace_count:
                    for trace in traces[last_trace_count:]:
                        if isinstance(trace, str):
                            yield {"event": "progress", "data": trace}
                        else:
                            yield {"event": "progress", "data": json.dumps(trace)}
                    last_trace_count = len(traces)
                    
                if job.status in ["completed", "failed"]:
                    yield {"event": "complete", "data": job.status}
                    break
                    
            await asyncio.sleep(1)
            
    return EventSourceResponse(event_generator())


@router.get("/{job_id}/report")
async def get_job_report(job_id: str):
    async with db.pool() as session:
        job = await session.get(JobModel, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        return {
            "job_id": job_id,
            "status": job.status,
            "report": job.metadata_.get("report", {}) if job.metadata_ else {}
        }

