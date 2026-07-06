from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.chunkers.base import Chunk
from src.core.embedders.embedder import Embedder, EmbeddedChunk


@pytest.mark.unit
class TestEmbedder:
    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_chunks_returns_embedded_chunks(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
        mock_litellm.aembedding = AsyncMock(return_value=mock_response)

        embedder = Embedder(model="test-model", batch_size=100)
        chunks = [
            Chunk(content="hello", index=0, metadata={"a": 1}),
            Chunk(content="world", index=1, metadata={"b": 2}),
        ]

        result = await embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert isinstance(result[0], EmbeddedChunk)
        assert result[0].content == "hello"
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[0].index == 0
        assert result[0].metadata == {"a": 1}
        assert result[1].content == "world"
        assert result[1].embedding == [0.4, 0.5, 0.6]

    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_chunks_empty_list(self, mock_litellm):
        embedder = Embedder()
        result = await embedder.embed_chunks([])
        assert result == []
        mock_litellm.aembedding.assert_not_called()

    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_chunks_calls_litellm_with_correct_model(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]
        mock_litellm.aembedding = AsyncMock(return_value=mock_response)

        embedder = Embedder(model="custom-model")
        chunks = [Chunk(content="test", index=0, metadata={})]

        await embedder.embed_chunks(chunks)

        mock_litellm.aembedding.assert_called_once_with(
            model="custom-model",
            input=["test"],
        )

    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_chunks_batches_correctly(self, mock_litellm):
        call_count = 0

        async def mock_aembedding(model, input):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.data = [{"embedding": [float(i)]} for i in range(len(input))]
            return mock_response

        mock_litellm.aembedding = mock_aembedding

        embedder = Embedder(model="test", batch_size=2)
        chunks = [
            Chunk(content="a", index=0, metadata={}),
            Chunk(content="b", index=1, metadata={}),
            Chunk(content="c", index=2, metadata={}),
        ]

        result = await embedder.embed_chunks(chunks)

        assert call_count == 2
        assert len(result) == 3

    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_query_returns_embedding(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_litellm.aembedding = AsyncMock(return_value=mock_response)

        embedder = Embedder(model="test-model")
        result = await embedder.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_litellm.aembedding.assert_called_once_with(
            model="test-model",
            input=["test query"],
        )

    @patch("src.core.embedders.embedder.litellm")
    async def test_embed_chunks_single_batch(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}, {"embedding": [0.2]}]
        mock_litellm.aembedding = AsyncMock(return_value=mock_response)

        embedder = Embedder(model="test", batch_size=100)
        chunks = [
            Chunk(content="x", index=0, metadata={}),
            Chunk(content="y", index=1, metadata={}),
        ]

        result = await embedder.embed_chunks(chunks)

        mock_litellm.aembedding.assert_called_once()
        assert len(result) == 2

    def test_default_model(self):
        embedder = Embedder()
        assert embedder._model_override is None

    def test_default_batch_size(self):
        embedder = Embedder()
        assert embedder.batch_size == 100
