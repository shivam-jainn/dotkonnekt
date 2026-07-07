"""Integration tests for the full Document IR chunk navigation flow.

Verifies: SemanticChunker → Pipeline → Worker → StorageWorker → Qdrant payload
and retrieval-side prev/next navigation.
"""

import json

import pytest

from src.core.document import Document, Heading, Page, Paragraph
from src.core.embedders.embedder import EmbeddedChunk
from src.core.chunkers.semantic import SemanticChunker
from src.models.storage import StorageMessage, StoredChunk


pytestmark = pytest.mark.integration


def _build_legal_doc() -> Document:
    return Document(
        filename="insurance_policy.pdf",
        metadata={"job_id": "test-nav-001"},
        pages=[
            Page(
                number=1,
                headings=[Heading("1. Definitions", 1, 1)],
                paragraphs=[
                    Paragraph(
                        '"Policyholder" means the person named in the Schedule.',
                        1,
                        "definition",
                    ),
                    Paragraph(
                        '"Insured Event" means any event giving rise to a claim.',
                        1,
                        "definition",
                    ),
                    Paragraph(
                        "The Insurer undertakes to pay the Sum Insured upon occurrence of an Insured Event.",
                        1,
                    ),
                ],
            ),
            Page(
                number=2,
                headings=[Heading("2. Obligations", 1, 2)],
                paragraphs=[
                    Paragraph(
                        "The Policyholder shall pay the premium upon Policy issuance.",
                        2,
                    ),
                    Paragraph(
                        "The Insurer shall not be liable for any indirect or consequential damages.",
                        2,
                    ),
                    Paragraph(
                        "Claims must be submitted within 30 days of the Insured Event.",
                        2,
                    ),
                ],
            ),
            Page(
                number=3,
                headings=[Heading("3. Exclusions", 1, 3)],
                paragraphs=[
                    Paragraph(
                        "This Policy does not cover war, nuclear hazards, or intentional acts.",
                        3,
                    ),
                    Paragraph(
                        "Pre-existing diseases are excluded during the first 48 months.",
                        3,
                    ),
                ],
            ),
        ],
    )


class TestChunkNavigationFlow:
    """Full flow: SemanticChunker → EmbeddedChunk → StoredChunk → Qdrant payload → retrieval."""

    def test_semantic_chunker_produces_linked_chunks(self):
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=120, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 4, "Should produce multiple chunks from multi-page doc"

        # First chunk has no previous
        assert chunks[0].previous_chunk_id is None
        # Last chunk has no next
        assert chunks[-1].next_chunk_id is None

        # Middle chunks are linked
        for i in range(1, len(chunks) - 1):
            assert chunks[i].previous_chunk_id is not None, f"Chunk {i} missing prev"
            assert chunks[i].next_chunk_id is not None, f"Chunk {i} missing next"

        # Links are consistent: chunk[i].next == chunk[i+1].id
        for i in range(len(chunks) - 1):
            assert chunks[i].next_chunk_id == chunks[i + 1].id, (
                f"Chunk {i} next={chunks[i].next_chunk_id} != chunk {i+1} id={chunks[i+1].id}"
            )
            assert chunks[i + 1].previous_chunk_id == chunks[i].id, (
                f"Chunk {i+1} prev={chunks[i+1].previous_chunk_id} != chunk {i} id={chunks[i].id}"
            )

    def test_links_survive_queue_transport(self):
        """Simulate: EmbeddedChunk → StoredChunk → JSON → StorageWorker → EmbeddedChunk."""
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=120, chunk_overlap=0)
        doc_chunks = chunker.chunk_document(doc)

        # Pipeline creates EmbeddedChunks
        embedded = [
            EmbeddedChunk(
                content=c.content,
                embedding=[0.1] * 5,
                index=c.index,
                metadata={**c.metadata, "job_id": "j1", "filename": "doc.pdf"},
                page=c.page,
                section=c.section,
                clause=c.clause,
                previous_chunk_id=c.previous_chunk_id,
                next_chunk_id=c.next_chunk_id,
            )
            for c in doc_chunks
        ]

        # Worker creates StorageMessage
        message = StorageMessage(
            job_id="j1",
            collection="test",
            chunks=[
                StoredChunk(
                    content=ec.content,
                    embedding=ec.embedding,
                    index=ec.index,
                    metadata={**ec.metadata, "job_id": "j1"},
                    page=ec.page,
                    section=ec.section,
                    clause=ec.clause,
                    previous_chunk=ec.previous_chunk_id,
                    next_chunk=ec.next_chunk_id,
                )
                for ec in embedded
            ],
        )

        # Queue transport (JSON serialize + deserialize)
        json_bytes = message.model_dump_json()
        msg_from_queue = StorageMessage.model_validate_json(json_bytes)

        # StorageWorker deserializes to EmbeddedChunk
        reconstructed = [
            EmbeddedChunk(
                content=c.content,
                embedding=c.embedding,
                index=c.index,
                metadata=c.metadata,
                page=c.page,
                section=c.section,
                clause=c.clause,
                previous_chunk_id=c.previous_chunk,
                next_chunk_id=c.next_chunk,
            )
            for c in msg_from_queue.chunks
        ]

        # Verify links survived
        assert len(reconstructed) == len(doc_chunks)
        for i in range(len(reconstructed)):
            orig = doc_chunks[i]
            recon = reconstructed[i]
            assert recon.previous_chunk_id == orig.previous_chunk_id, (
                f"Chunk {i}: prev lost during transport"
            )
            assert recon.next_chunk_id == orig.next_chunk_id, (
                f"Chunk {i}: next lost during transport"
            )

    def test_qdrant_payload_has_navigation_fields(self):
        """Verify to_payload() includes prev/next for Qdrant storage."""
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=120, chunk_overlap=0)
        doc_chunks = chunker.chunk_document(doc)

        embedded = [
            EmbeddedChunk(
                content=c.content,
                embedding=[0.1] * 5,
                index=c.index,
                metadata={**c.metadata, "job_id": "j1", "filename": "doc.pdf"},
                page=c.page,
                section=c.section,
                clause=c.clause,
                previous_chunk_id=c.previous_chunk_id,
                next_chunk_id=c.next_chunk_id,
            )
            for c in doc_chunks
        ]

        for i, ec in enumerate(embedded):
            payload = ec.to_payload()
            if i == 0:
                assert payload["previous_chunk"] is None
                assert payload["next_chunk"] is not None
            elif i == len(embedded) - 1:
                assert payload["previous_chunk"] is not None
                assert payload["next_chunk"] is None
            else:
                assert payload["previous_chunk"] is not None
                assert payload["next_chunk"] is not None

            # Navigation fields present
            assert "page" in payload
            assert "section" in payload

    def test_retrieval_response_includes_navigation(self):
        """Simulate QA retrieval returning chunks with navigation fields."""
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=120, chunk_overlap=0)
        doc_chunks = chunker.chunk_document(doc)

        embedded = [
            EmbeddedChunk(
                content=c.content,
                embedding=[0.1] * 5,
                index=c.index,
                metadata={**c.metadata, "job_id": "j1", "filename": "doc.pdf"},
                page=c.page,
                section=c.section,
                clause=c.clause,
                previous_chunk_id=c.previous_chunk_id,
                next_chunk_id=c.next_chunk_id,
            )
            for c in doc_chunks
        ]

        # Simulate what retrieve_context_node returns
        retrieved = [
            {
                "content": ec.content,
                "index": ec.index,
                "score": 0.95,
                "page": ec.page,
                "section": ec.section,
                "clause": ec.clause,
                "previous_chunk": ec.previous_chunk_id,
                "next_chunk": ec.next_chunk_id,
            }
            for ec in embedded
        ]

        # Verify navigation is available for hop
        for i, chunk in enumerate(retrieved):
            if i > 0:
                assert chunk["previous_chunk"] is not None
            if i < len(retrieved) - 1:
                assert chunk["next_chunk"] is not None

        # Simulate "hop next" from chunk 0
        first = retrieved[0]
        assert first["next_chunk"] is not None

        # Simulate navigation map
        nav_map = {c["index"]: c for c in retrieved}
        current = nav_map[0]
        visited = [0]

        while current["next_chunk"]:
            next_idx = current["index"] + 1
            current = nav_map[next_idx]
            visited.append(current["index"])

        assert visited == list(range(len(retrieved))), "Should traverse all chunks via next"

    def test_chunks_have_section_metadata_for_context(self):
        """Each chunk should carry section info for enriched retrieval."""
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=120, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        sections_seen = set()
        for c in chunks:
            if c.section:
                sections_seen.add(c.section)

        # Should detect our 3 sections
        assert "1. Definitions" in sections_seen
        assert "2. Obligations" in sections_seen
        assert "3. Exclusions" in sections_seen

    def test_clause_classification_preserved(self):
        """Chunks should carry clause kind for legal analysis."""
        doc = _build_legal_doc()
        chunker = SemanticChunker(max_chunk_size=80, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        # At least some chunks should have clause labels (numbered sections like "1.", "2.")
        clauses_detected = [c.clause for c in chunks if c.clause is not None]
        assert len(clauses_detected) > 0, f"Should detect clause labels, got: {[c.clause for c in chunks]}"
