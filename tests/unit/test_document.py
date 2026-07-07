import pytest

from src.core.document import (
    Block,
    Chunk,
    Clause,
    ClauseEdge,
    ClauseGraph,
    Document,
    Entity,
    ExtractedMetadata,
    Heading,
    Image,
    Page,
    Paragraph,
    SemanticMetadata,
    Section,
    Table,
)


@pytest.mark.unit
class TestDocumentIR:
    def test_document_creation(self):
        doc = Document(filename="test.pdf")
        assert doc.filename == "test.pdf"
        assert doc.id is not None
        assert doc.pages == []
        assert doc.chunks == []

    def test_document_total_pages(self):
        doc = Document(
            filename="test.pdf",
            pages=[Page(number=1), Page(number=2), Page(number=3)],
        )
        assert doc.total_pages == 3

    def test_document_get_page_text(self):
        doc = Document(
            filename="test.pdf",
            pages=[
                Page(number=1, text="Page 1 content"),
                Page(number=2, text="Page 2 content"),
            ],
        )
        assert doc.get_page_text(1) == "Page 1 content"
        assert doc.get_page_text(2) == "Page 2 content"
        assert doc.get_page_text(99) == ""

    def test_chunk_to_payload(self):
        chunk = Chunk(
            id="c1",
            content="Test content",
            index=0,
            page=1,
            section="Section 1",
            clause="1.1",
        )
        payload = chunk.to_payload()
        assert payload["content"] == "Test content"
        assert payload["page"] == 1
        assert payload["section"] == "Section 1"
        assert payload["clause"] == "1.1"

    def test_chunk_to_payload_with_semantic_metadata(self):
        chunk = Chunk(
            id="c1",
            content="Test",
            index=0,
            semantic_metadata=SemanticMetadata(
                summary="A test chunk",
                keywords=["test", "chunk"],
            ),
        )
        payload = chunk.to_payload()
        assert payload["summary"] == "A test chunk"
        assert payload["keywords"] == ["test", "chunk"]

    def test_chunk_to_payload_backward_compat(self):
        chunk = Chunk(
            id="c1",
            content="Test",
            index=0,
            metadata={"job_id": "job-1", "filename": "test.pdf"},
        )
        payload = chunk.to_payload()
        assert "job_id" in payload
        assert "filename" in payload

    def test_clause_graph_operations(self):
        graph = ClauseGraph()
        clause1 = Clause(id="c1", text="First clause")
        clause2 = Clause(id="c2", text="Second clause")
        clause3 = Clause(id="c3", text="Third clause")

        graph.add_clause(clause1)
        graph.add_clause(clause2)
        graph.add_clause(clause3)

        graph.link_chain(["c1", "c2", "c3"])

        assert graph.edges["c1"].next_clause == "c2"
        assert graph.edges["c2"].previous_clause == "c1"
        assert graph.edges["c2"].next_clause == "c3"
        assert graph.edges["c3"].previous_clause == "c2"

    def test_entity_creation(self):
        entity = Entity(name="Acme Corp", kind="organization")
        assert entity.name == "Acme Corp"
        assert entity.kind == "organization"

    def test_extracted_metadata_defaults(self):
        meta = ExtractedMetadata()
        assert meta.document_type is None
        assert meta.entities == []
        assert meta.obligations == []

    def test_semantic_metadata_defaults(self):
        meta = SemanticMetadata()
        assert meta.summary is None
        assert meta.keywords == []

    def test_page_with_blocks(self):
        page = Page(
            number=1,
            blocks=[
                Block(kind="heading", content="Title", level=1, page=1),
                Block(kind="paragraph", content="Body text", page=1),
            ],
        )
        assert len(page.blocks) == 2
        assert page.blocks[0].kind == "heading"
        assert page.blocks[1].kind == "paragraph"

    def test_heading_creation(self):
        heading = Heading(text="Section 1", level=1, page=1)
        assert heading.text == "Section 1"
        assert heading.level == 1

    def test_paragraph_creation(self):
        para = Paragraph(text="Body text", page=1, kind="body")
        assert para.text == "Body text"
        assert para.kind == "body"

    def test_image_creation(self):
        img = Image(bytes=b"fake", ext="png", page=1, index=0)
        assert img.bytes == b"fake"
        assert img.ext == "png"
