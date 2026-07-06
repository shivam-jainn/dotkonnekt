import pytest

from src.core.chunkers.text import TextChunker


pytestmark = pytest.mark.integration


class TestTextChunkerIntegration:
    def test_chunk_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)

        result = chunker.chunk(text)

        assert len(result) >= 2
        all_content = " ".join(c.content for c in result)
        assert "First paragraph" in all_content
        assert "Second paragraph" in all_content
        assert "Third paragraph" in all_content

    def test_chunk_long_document(self):
        paragraphs = [f"Paragraph {i} with some content here." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        result = chunker.chunk(text)

        assert len(result) > 5
        for chunk in result:
            assert len(chunk.content) <= 120  # some tolerance for overlap

    def test_chunk_sentences(self):
        text = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        )
        chunker = TextChunker(chunk_size=40, chunk_overlap=0)

        result = chunker.chunk(text)

        assert len(result) >= 2
        for chunk in result:
            assert len(chunk.content) <= 50  # tolerance

    def test_chunk_preserves_order(self):
        text = "Alpha\n\nBeta\n\nGamma\n\nDelta"
        chunker = TextChunker(chunk_size=20, chunk_overlap=0)

        result = chunker.chunk(text)

        indices = [c.index for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_large_paragraph(self):
        text = "word " * 500  # 2500 chars
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        result = chunker.chunk(text)

        assert len(result) > 1
        all_content = " ".join(c.content for c in result)
        assert "word" in all_content

    def test_chunk_metadata_propagation(self):
        text = "Hello\n\nWorld"
        chunker = TextChunker(chunk_size=10, chunk_overlap=0)

        result = chunker.chunk(text, metadata={"source": "test", "doc_id": "42"})

        for chunk in result:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["doc_id"] == "42"
            assert "chunk_index" in chunk.metadata
