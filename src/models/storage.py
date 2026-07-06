from pydantic import BaseModel


class StoredChunk(BaseModel):
    content: str
    embedding: list[float]
    index: int
    metadata: dict


class StorageMessage(BaseModel):
    job_id: str
    collection: str
    chunks: list[StoredChunk]
