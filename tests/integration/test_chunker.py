import pytest

from src.core.chunkers.semantic import SemanticChunker, TextChunker
from src.core.document import Document, Heading, Page, Paragraph


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
            assert "page" in chunk.metadata


class TestSemanticChunkerIntegration:
    def test_chunk_legal_document(self):
        doc = Document(
            filename="contract.pdf",
            pages=[
                Page(
                    number=1,
                    text=(
                        "SERVICE AGREEMENT\n\n"
                        "1. Scope of Services. The Provider shall deliver software development services.\n\n"
                        "2. Payment Terms. The Client shall pay $10,000 within 30 days of invoice.\n\n"
                        "3. Confidentiality. The Provider shall not disclose Confidential Information."
                    ),
                )
            ],
        )
        chunker = SemanticChunker(max_chunk_size=200)
        result = chunker.chunk_document(doc)

        assert len(result) >= 3
        # Should detect numbered clauses
        clauses = [c for c in result if c.clause is not None]
        assert len(clauses) >= 2

    def test_chunk_preserves_sections(self):
        doc = Document(
            filename="doc.pdf",
            pages=[
                Page(
                    number=1,
                    headings=[Heading(text="Introduction", level=1, page=1)],
                    paragraphs=[Paragraph(text="Welcome to the document.", page=1)],
                ),
                Page(
                    number=2,
                    headings=[Heading(text="Details", level=1, page=2)],
                    paragraphs=[Paragraph(text="Here are the details.", page=2)],
                ),
            ],
        )
        chunker = SemanticChunker(max_chunk_size=500)
        result = chunker.chunk_document(doc)

        sections = {c.section for c in result if c.section}
        assert len(sections) >= 2

    def test_chunk_with_large_clause_splits(self):
        doc = Document(
            filename="long.pdf",
            pages=[
                Page(
                    number=1,
                    text="1. " + "This is a very long clause. " * 100,
                )
            ],
        )
        chunker = SemanticChunker(max_chunk_size=200)
        result = chunker.chunk_document(doc)

        # Should split the large clause into multiple chunks
        assert len(result) > 1

    def test_chunk_page_tracking(self):
        doc = Document(
            filename="multi.pdf",
            pages=[
                Page(number=1, text="Page 1 content."),
                Page(number=2, text="Page 2 content."),
                Page(number=3, text="Page 3 content."),
            ],
        )
        chunker = SemanticChunker(max_chunk_size=100)
        result = chunker.chunk_document(doc)

        pages_seen = {c.page for c in result}
        assert 1 in pages_seen
        assert 2 in pages_seen
        assert 3 in pages_seen
