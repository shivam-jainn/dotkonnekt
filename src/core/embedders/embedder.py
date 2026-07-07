from dataclasses import dataclass

import litellm

from src.core.chunkers.base import Chunk as LegacyChunk
from src.core.document import Chunk as DocChunk
from src.core.models.providers import TaskType
from src.core.models.registry import registry


@dataclass
class EmbeddedChunk:
    content: str
    embedding: list[float]
    index: int
    metadata: dict

    # --- Document IR fields (new) ---
    id: str = ""  # Document IR chunk ID — used as Qdrant point ID for navigation
    page: int = 0
    section: str | None = None
    subsection: str | None = None
    clause: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    clause_id: str | None = None
    semantic_metadata: dict | None = None
    content_type: str | None = None

    def to_payload(self) -> dict:
        """Serialize to a flat dict for Qdrant payload storage."""
        payload: dict = {
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "subsection": self.subsection,
            "clause": self.clause,
            "previous_chunk": self.previous_chunk_id,
            "next_chunk": self.next_chunk_id,
            "content_type": self.content_type,
            **self.metadata,
        }
        if self.clause_id:
            payload["clause_id"] = self.clause_id
        if self.semantic_metadata:
            sm = self.semantic_metadata
            for key in (
                "summary", "keywords", "obligations", "risks", "entities", "topics", 
                "deadlines", "rights", "exclusions", "definitions", "parties", 
                "jurisdictions", "document_type", "party_sentences", "obligations_by_party",
                "risks_by_party"
            ):
                val = sm.get(key) if isinstance(sm, dict) else getattr(sm, key, None)
                if val:
                    payload[key] = val
        return payload


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self._model_override = model
        self._api_base_override = api_base
        self._api_key_override = api_key
        self.batch_size = batch_size

    def _get_litellm_kwargs(self) -> dict:
        if self._model_override:
            kwargs: dict = {"model": self._model_override}
            if self._api_base_override:
                kwargs["api_base"] = self._api_base_override
            if self._api_key_override:
                kwargs["api_key"] = self._api_key_override
            return kwargs

        return registry.get_litellm_kwargs(TaskType.EMBEDDING)

    # ------------------------------------------------------------------
    # Legacy interface: list[LegacyChunk] -> list[EmbeddedChunk]
    # ------------------------------------------------------------------

    async def embed_chunks(self, chunks: list[LegacyChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        kwargs = self._get_litellm_kwargs()
        if not kwargs:
            raise RuntimeError(
                "No embedding model configured. "
                "Use PUT /api/v1/models/config to select an embedding model."
            )

        import asyncio

        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        async def _embed_batch(batch: list[LegacyChunk]) -> list[dict]:
            texts = [c.content for c in batch]
            response = await litellm.aembedding(**kwargs, input=texts)
            return response.data

        tasks = [_embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        all_embedded: list[EmbeddedChunk] = []
        for batch, embeddings in zip(batches, results):
            for chunk, embedding_data in zip(batch, embeddings):
                all_embedded.append(
                    EmbeddedChunk(
                        content=chunk.content,
                        embedding=embedding_data["embedding"],
                        index=chunk.index,
                        metadata=chunk.metadata,
                    )
                )

        return all_embedded

    # ------------------------------------------------------------------
    # Document IR interface: list[DocChunk] -> (list[DocChunk], list[list[float]])
    # ------------------------------------------------------------------

    async def embed_document_chunks(
        self, chunks: list[DocChunk]
    ) -> tuple[list[DocChunk], list[list[float]]]:
        """Embed Document IR chunks, returning chunks and their embeddings."""
        if not chunks:
            return [], []

        kwargs = self._get_litellm_kwargs()
        if not kwargs:
            raise RuntimeError(
                "No embedding model configured. "
                "Use PUT /api/v1/models/config to select an embedding model."
            )

        import asyncio

        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        async def _embed_batch(batch: list[DocChunk]) -> list[dict]:
            texts = [c.content for c in batch]
            response = await litellm.aembedding(**kwargs, input=texts)
            return response.data

        tasks = [_embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        all_embeddings: list[list[float]] = []
        for embeddings in results:
            for emb_data in embeddings:
                all_embeddings.append(emb_data["embedding"])

        return chunks, all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        kwargs = self._get_litellm_kwargs()
        if not kwargs:
            raise RuntimeError("No embedding model configured.")

        response = await litellm.aembedding(**kwargs, input=[text])
        return response.data[0]["embedding"]
