from typing import Dict, List, Optional
from pydantic import BaseModel


class StoredChunk(BaseModel):
    """Chunk payload sent to the storage queue.

    Maintains backward compat (content, index, metadata) while adding the
    new Document IR fields.
    """

    content: str
    embedding: list[float]
    index: int
    metadata: dict

    # --- Document IR fields (new) ---
    id: str = ""  # Document IR chunk ID — used as Qdrant point ID
    page: int = 0
    section: str | None = None
    subsection: str | None = None
    clause: str | None = None
    previous_chunk: str | None = None
    next_chunk: str | None = None
    summary: str | None = None
    keywords: list[str] = []
    obligations: list[str] = []
    entities: list[str] = []
    risks: list[str] = []
    document_type: str | None = None
    party_sentences: Dict[str, List[str]] = {}
    obligations_by_party: List[dict] = []
    risks_by_party: List[dict] = []


class StorageMessage(BaseModel):
    job_id: str
    collection: str
    chunks: list[StoredChunk]
