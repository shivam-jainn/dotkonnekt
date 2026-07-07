import pytest

from src.core.chunkers.text import TextChunker
from src.core.chunkers.semantic import SemanticChunker
from src.core.document import Document, Page, Heading, Paragraph


@pytest.mark.unit
class TestTextChunker:
    """Tests for the legacy TextChunker adapter (wraps SemanticChunker)."""

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
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_custom_separators(self):
        chunker = TextChunker(separators=["---", " "])
        assert chunker.separators == ["---", " "]


@pytest.mark.unit
class TestSemanticChunker:
    """Tests for the new SemanticChunker that operates on Document IR."""

    def test_chunk_document_empty(self):
        doc = Document(filename="empty.pdf")
        chunker = SemanticChunker(max_chunk_size=1000)
        result = chunker.chunk_document(doc)
        assert result == []

    def test_chunk_document_single_page(self):
        doc = Document(
            filename="test.pdf",
            pages=[
                Page(
                    number=1,
                    text="Hello world. This is a test document.",
                )
            ],
        )
        chunker = SemanticChunker(max_chunk_size=1000)
        result = chunker.chunk_document(doc)
        assert len(result) >= 1
        assert "Hello world" in result[0].content

    def test_chunk_document_with_headings(self):
        doc = Document(
            filename="structured.pdf",
            pages=[
                Page(
                    number=1,
                    headings=[
                        Heading(text="Introduction", level=1, page=1),
                    ],
                    paragraphs=[
                        Paragraph(text="This is the introduction.", page=1),
                    ],
                ),
                Page(
                    number=2,
                    headings=[
                        Heading(text="Section 2", level=2, page=2),
                    ],
                    paragraphs=[
                        Paragraph(text="This is section 2.", page=2),
                    ],
                ),
            ],
        )
        chunker = SemanticChunker(max_chunk_size=1000)
        result = chunker.chunk_document(doc)
        assert len(result) >= 2
        # First chunk should contain the introduction
        assert any("Introduction" in c.content or "introduction" in c.content for c in result)
        # Second chunk should contain section 2
        assert any("section 2" in c.content.lower() for c in result)

    def test_chunk_document_preserves_page_info(self):
        doc = Document(
            filename="pages.pdf",
            pages=[
                Page(number=1, text="Page one content."),
                Page(number=2, text="Page two content."),
            ],
        )
        chunker = SemanticChunker(max_chunk_size=1000)
        result = chunker.chunk_document(doc)
        pages_seen = {c.page for c in result}
        assert 1 in pages_seen
        assert 2 in pages_seen

    def test_chunk_document_metadata_has_job_id(self):
        doc = Document(
            filename="test.pdf",
            metadata={"job_id": "job-123"},
            pages=[Page(number=1, text="Some content.")],
        )
        chunker = SemanticChunker(max_chunk_size=1000)
        result = chunker.chunk_document(doc)
        assert len(result) >= 1
        assert result[0].metadata.get("job_id") == "job-123"
        assert result[0].metadata.get("filename") == "test.pdf"

    def test_chunk_document_prev_next_links(self):
        doc = Document(
            filename="test.pdf",
            pages=[Page(number=1, text="A\n\nB\n\nC\n\nD\n\nE\n\nF\n\nG\n\nH\n\nI\n\nJ")],
        )
        chunker = SemanticChunker(max_chunk_size=20, chunk_overlap=0)
        result = chunker.chunk_document(doc)
        if len(result) >= 3:
            # Second chunk should have a previous link
            assert result[1].previous_chunk_id == result[0].id
            # Second chunk should be the next of the first
            assert result[0].next_chunk_id == result[1].id

    def test_chunk_document_clause_detection(self):
        doc = Document(
            filename="contract.pdf",
            pages=[
                Page(
                    number=1,
                    text="1. First obligation. The party shall do something.\n\n2. Second obligation. The party shall do another thing.",
                )
            ],
        )
        chunker = SemanticChunker(max_chunk_size=500)
        result = chunker.chunk_document(doc)
        # Should detect numbered clauses
        clauses_detected = [c for c in result if c.clause is not None]
        assert len(clauses_detected) >= 1
