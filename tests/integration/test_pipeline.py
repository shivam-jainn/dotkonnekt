import os
from unittest.mock import MagicMock

import pytest

from src.core.parsers.pdf import PDFParser
from src.core.pipeline import Pipeline, PipelineResult


pytestmark = pytest.mark.integration


def _make_pdf(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    return storage


class TestPipelineIntegration:
    def test_parse_pdf_directly(self):
        pdf_data = _make_pdf("Integration test content for pipeline.")
        parser = PDFParser()

        result = parser.parse(pdf_data, "pipeline.pdf")

        assert "Integration test content" in result.content
        assert result.metadata["parser"] == "pdf"

    def test_chunk_parsed_document(self):
        from src.core.chunkers.text import TextChunker

        text = "First section.\n\nSecond section.\n\nThird section."
        chunker = TextChunker(chunk_size=30, chunk_overlap=0)

        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        all_content = " ".join(c.content for c in chunks)
        assert "First section" in all_content

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_full_pipeline_parse_chunk_embed(self, mock_storage):
        pdf_data = _make_pdf(
            "This is a test document for the full pipeline integration test."
        )
        mock_storage.download_bytes.return_value = pdf_data

        pipeline = Pipeline()
        pipeline.storage = mock_storage

        result = await pipeline.run(
            job_id="integration-test",
            files=[
                {
                    "filename": "test.pdf",
                    "storage_path": "integration-test/test.pdf",
                }
            ],
        )

        assert isinstance(result, PipelineResult)
        assert result.job_id == "integration-test"
        assert result.total_chunks > 0
        assert len(result.embedded_chunks) > 0
        assert result.errors == []

        for embedded in result.embedded_chunks:
            assert len(embedded.embedding) > 0
            assert embedded.content

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_full_pipeline_multiple_files(self, mock_storage):
        pdf1 = _make_pdf("Document one content.")
        pdf2 = _make_pdf("Document two content.")

        def download_side_effect(path):
            if "doc1" in path:
                return pdf1
            return pdf2

        mock_storage.download_bytes.side_effect = download_side_effect

        pipeline = Pipeline()
        pipeline.storage = mock_storage

        result = await pipeline.run(
            job_id="multi-test",
            files=[
                {"filename": "doc1.pdf", "storage_path": "multi-test/doc1.pdf"},
                {"filename": "doc2.pdf", "storage_path": "multi-test/doc2.pdf"},
            ],
        )

        assert result.total_chunks > 0
        assert len(result.embedded_chunks) > 0
        assert result.errors == []
