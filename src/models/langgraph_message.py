from pydantic import BaseModel


class RawChunk(BaseModel):
    """A single text chunk with its metadata, before any LLM analysis."""

    content: str
    index: int
    metadata: dict


class LangGraphMessage(BaseModel):
    """
    Message published to the langgraph queue by the ingestion worker.

    Carries the raw (un-analysed) chunks for a completed ingestion job.
    The LangGraphWorker consumes this and runs the LangGraph agent graph
    to produce structured findings (obligations, entities, risky terms, etc.)
    """

    job_id: str
    collection: str
    chunks: list[RawChunk]
