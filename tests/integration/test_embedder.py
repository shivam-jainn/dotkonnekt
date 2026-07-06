import os

import pytest

from src.core.chunkers.base import Chunk
from src.core.embedders.embedder import Embedder, EmbeddedChunk


pytestmark = pytest.mark.integration


@pytest.fixture
def requires_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping embedding tests")


class TestEmbedderIntegration:
    async def test_embed_single_chunk(self, requires_api_key):
        embedder = Embedder(model="text-embedding-3-small", batch_size=10)
        chunks = [Chunk(content="Hello world", index=0, metadata={})]

        result = await embedder.embed_chunks(chunks)

        assert len(result) == 1
        assert isinstance(result[0], EmbeddedChunk)
        assert result[0].content == "Hello world"
        assert len(result[0].embedding) > 0
        assert all(isinstance(v, float) for v in result[0].embedding)

    async def test_embed_multiple_chunks(self, requires_api_key):
        embedder = Embedder(model="text-embedding-3-small", batch_size=10)
        chunks = [
            Chunk(content="Machine learning", index=0, metadata={}),
            Chunk(content="Deep learning", index=1, metadata={}),
            Chunk(content="Natural language processing", index=2, metadata={}),
        ]

        result = await embedder.embed_chunks(chunks)

        assert len(result) == 3
        for r in result:
            assert len(r.embedding) > 0

    async def test_embed_query(self, requires_api_key):
        embedder = Embedder(model="text-embedding-3-small")

        embedding = await embedder.embed_query("What is AI?")

        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    async def test_embeddings_are_normalized_directionally(self, requires_api_key):
        embedder = Embedder(model="text-embedding-3-small", batch_size=10)
        chunks = [
            Chunk(content="cat", index=0, metadata={}),
            Chunk(content="dog", index=1, metadata={}),
            Chunk(content="automobile", index=2, metadata={}),
        ]

        result = await embedder.embed_chunks(chunks)

        import math

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def mag(a):
            return math.sqrt(sum(x * x for x in a))

        cat_dog = dot(result[0].embedding, result[1].embedding) / (
            mag(result[0].embedding) * mag(result[1].embedding)
        )
        cat_car = dot(result[0].embedding, result[2].embedding) / (
            mag(result[0].embedding) * mag(result[2].embedding)
        )

        assert cat_dog > cat_car

    async def test_embed_empty_list(self, requires_api_key):
        embedder = Embedder(model="text-embedding-3-small")
        result = await embedder.embed_chunks([])
        assert result == []
