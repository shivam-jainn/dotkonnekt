import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.database import db
from src.database.models import JobModel

router = APIRouter(prefix="/query", tags=["Query"])
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="Natural-language question to answer from the document")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve from the vector store")
    collection: str | None = Field(None, description="Override the Qdrant collection to search in (defaults to settings.qdrant_collection)")


class ContextChunk(BaseModel):
    index: int
    content: str
    score: float | None = None


class QueryResponse(BaseModel):
    job_id: str
    query: str
    answer: str
    context_chunks: list[ContextChunk]
    collection: str


async def _assert_job_exists(job_id: str) -> JobModel:
    """Raise 404 if the job does not exist, 400 if it isn't completed yet."""
    async with db.pool() as session:
        job = await session.get(JobModel, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if job.status not in ("completed",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' is not ready for querying (status: {job.status}). "
                   "Wait until ingestion and analysis are complete.",
        )
    return job


@router.post(
    "/{job_id}",
    response_model=QueryResponse,
    summary="Query a document using RAG",
    description=(
        "Embeds the user query, retrieves the top-K most relevant chunks from the "
        "vector store (filtered by job_id), and generates a grounded answer using the "
        "configured LLM. The job must be in **completed** status."
    ),
)
async def query_document(job_id: str, request: QueryRequest) -> QueryResponse:
    await _assert_job_exists(job_id)

    from src.core.agent.qa import run_qa
    from src.configs import settings

    collection = request.collection or settings.qdrant_collection

    try:
        result = await run_qa(
            job_id=job_id,
            query=request.query,
            top_k=request.top_k,
            collection=collection,
        )

        print(f"[query] QA result for job {job_id}: {result}")
    except Exception as exc:
        logger.exception("[query] QA pipeline failed for job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(exc)}",
        )

    return QueryResponse(
        job_id=job_id,
        query=request.query,
        answer=result["answer"],
        context_chunks=[
            ContextChunk(
                index=c.get("index", 0),
                content=c.get("content", ""),
                score=c.get("score"),
            )
            for c in result.get("context_chunks", [])
        ],
        collection=collection,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------


@router.post(
    "/{job_id}/stream",
    summary="Stream a document query answer via SSE",
    description=(
        "Same as POST /{job_id} but streams the LLM answer token-by-token as "
        "Server-Sent Events. Emits `context` events with the retrieved chunks "
        "first, then `token` events for each answer token, and a final `done` "
        "event with the complete answer. "
        "The job must be in **completed** status."
    ),
)
async def stream_query(job_id: str, request: QueryRequest):
    await _assert_job_exists(job_id)

    from src.core.agent.qa import run_qa_stream
    from src.configs import settings

    collection = request.collection or settings.qdrant_collection

    async def event_generator() -> AsyncIterator[dict]:
        try:
            async for event in run_qa_stream(
                job_id=job_id,
                query=request.query,
                top_k=request.top_k,
                collection=collection,
            ):
                yield event
        except Exception as exc:
            logger.exception("[query/stream] Streaming QA failed for job %s", job_id)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}),
            }

    return EventSourceResponse(event_generator())
