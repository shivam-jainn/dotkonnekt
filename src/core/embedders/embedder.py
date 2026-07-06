from dataclasses import dataclass

import litellm

from src.core.chunkers.base import Chunk
from src.core.models.providers import TaskType
from src.core.models.registry import registry


@dataclass
class EmbeddedChunk:
    content: str
    embedding: list[float]
    index: int
    metadata: dict


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

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        kwargs = self._get_litellm_kwargs()
        if not kwargs:
            raise RuntimeError(
                "No embedding model configured. "
                "Use PUT /api/v1/models/config to select an embedding model."
            )

        all_embedded: list[EmbeddedChunk] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.content for c in batch]

            response = await litellm.aembedding(**kwargs, input=texts)

            embeddings = response.data

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

    async def embed_query(self, text: str) -> list[float]:
        kwargs = self._get_litellm_kwargs()
        if not kwargs:
            raise RuntimeError("No embedding model configured.")

        response = await litellm.aembedding(**kwargs, input=[text])
        return response.data[0]["embedding"]
