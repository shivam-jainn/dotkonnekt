import pytest

from src.core.document import (
    Chunk,
    Clause,
    ClauseEdge,
    ClauseGraph,
    Document,
    Heading,
    Page,
    Paragraph,
    Section,
)
from src.core.clause_graph import build_clause_graph


@pytest.mark.unit
class TestClauseGraph:
    def test_empty_document(self):
        doc = Document(filename="empty.pdf")
        graph = build_clause_graph(doc)
        assert len(graph.clauses) == 0

    def test_chunks_without_clauses(self):
        doc = Document(
            filename="simple.pdf",
            chunks=[
                Chunk(id="c1", content="Hello world", index=0, page=1),
                Chunk(id="c2", content="Goodbye world", index=1, page=1),
            ],
        )
        graph = build_clause_graph(doc)
        assert len(graph.clauses) == 0

    def test_chunks_with_clauses(self):
        doc = Document(
            filename="contract.pdf",
            chunks=[
                Chunk(
                    id="c1",
                    content="1. The party shall pay rent.",
                    index=0,
                    page=1,
                    clause="1.",
                    section="Terms",
                ),
                Chunk(
                    id="c2",
                    content="2. The party shall maintain the property.",
                    index=1,
                    page=1,
                    clause="2.",
                    section="Terms",
                ),
            ],
        )
        graph = build_clause_graph(doc)
        assert len(graph.clauses) == 2

    def test_clause_chaining(self):
        doc = Document(
            filename="contract.pdf",
            chunks=[
                Chunk(id="c1", content="1. First.", index=0, page=1, clause="1."),
                Chunk(id="c2", content="2. Second.", index=1, page=1, clause="2."),
                Chunk(id="c3", content="3. Third.", index=2, page=1, clause="3."),
            ],
        )
        graph = build_clause_graph(doc)
        # Check chaining
        clause_ids = list(graph.clauses.keys())
        if len(clause_ids) >= 3:
            assert graph.edges[clause_ids[1]].previous_clause == clause_ids[0]
            assert graph.edges[clause_ids[0]].next_clause == clause_ids[1]
            assert graph.edges[clause_ids[2]].previous_clause == clause_ids[1]

    def test_clause_kind_classification(self):
        doc = Document(
            filename="contract.pdf",
            chunks=[
                Chunk(
                    id="c1",
                    content="The party shall deliver the goods within 30 days.",
                    index=0,
                    page=1,
                    clause="1.",
                ),
                Chunk(
                    id="c2",
                    content='"Goods" means the products listed in Exhibit A.',
                    index=1,
                    page=1,
                    clause="2.",
                ),
                Chunk(
                    id="c3",
                    content="The Client may terminate with 30 days notice.",
                    index=2,
                    page=1,
                    clause="3.",
                ),
                Chunk(
                    id="c4",
                    content="The Provider shall not be liable for indirect damages.",
                    index=3,
                    page=1,
                    clause="4.",
                ),
            ],
        )
        graph = build_clause_graph(doc)
        kinds = {c.kind for c in graph.clauses.values()}
        assert "obligation" in kinds
        assert "definition" in kinds
        assert "right" in kinds
        assert "exclusion" in kinds

    def test_chunks_get_clause_id_assigned(self):
        doc = Document(
            filename="test.pdf",
            chunks=[
                Chunk(id="c1", content="1. First.", index=0, page=1, clause="1."),
            ],
        )
        build_clause_graph(doc)
        assert doc.chunks[0].clause_id is not None
