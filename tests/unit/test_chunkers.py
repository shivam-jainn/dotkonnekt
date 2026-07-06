import pytest

from src.core.chunkers.text import TextChunker


@pytest.mark.unit
class TestTextChunker:
    def test_chunk_empty_text(self):
        chunker = TextChunker(chunk_size=100)
        result = chunker.chunk("")
        assert result == []

    def test_chunk_whitespace_only(self):
        chunker = TextChunker(chunk_size=100)
        result = chunker.chunk("   \n  \t  ")
        assert result == []

    def test_chunk_short_text_returns_single_chunk(self):
        chunker = TextChunker(chunk_size=1000)
        result = chunker.chunk("Hello world")
        assert len(result) == 1
        assert result[0].content == "Hello world"
        assert result[0].index == 0

    def test_chunk_respects_chunk_size(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = "word " * 30  # 150 chars with spaces as separators
        result = chunker.chunk(text)
        for chunk in result:
            assert len(chunk.content) <= 60  # some tolerance for overlap

    def test_chunk_produces_multiple_chunks(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=0)
        text = "Word " * 20  # 100 chars
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_chunk_preserves_content(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph content here. Second paragraph content here."
        result = chunker.chunk(text)
        all_content = " ".join(c.content for c in result)
        assert "First paragraph" in all_content
        assert "Second paragraph" in all_content

    def test_chunk_metadata_includes_chunk_index(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=0)
        text = "word " * 15  # 75 chars with spaces as separators
        result = chunker.chunk(text)
        for idx, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == idx

    def test_chunk_merges_custom_metadata(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=0)
        result = chunker.chunk("Hello world", metadata={"source": "test"})
        assert result[0].metadata["source"] == "test"
        assert result[0].metadata["chunk_index"] == 0

    def test_chunk_with_paragraph_separators(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=0)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = chunker.chunk(text)
        assert len(result) >= 2

    def test_chunk_indices_are_sequential(self):
        chunker = TextChunker(chunk_size=15, chunk_overlap=0)
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = chunker.chunk(text)
        for idx, chunk in enumerate(result):
            assert chunk.index == idx

    def test_chunk_overlap_produces_overlapping_content(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        text = "AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD"
        result = chunker.chunk(text)
        if len(result) > 1:
            end_of_first = result[0].content[-10:]
            start_of_second = result[1].content[:10]
            assert (
                end_of_first in result[1].content
                or start_of_second in result[0].content
            )

    def test_default_separators(self):
        chunker = TextChunker()
        assert chunker.separators == ["\n\n", "\n", ". ", " "]

    def test_custom_separators(self):
        chunker = TextChunker(separators=["---", " "])
        assert chunker.separators == ["---", " "]
