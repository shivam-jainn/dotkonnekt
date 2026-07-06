from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.embedders.embedder import EmbeddedChunk
from src.core.pipeline import Pipeline, PipelineResult


@pytest.mark.unit
class TestPipeline:
    @patch("src.core.pipeline.create_storage")
    def test_get_parser_pdf(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        pipeline = Pipeline()
        parser = pipeline._get_parser("document.pdf")
        assert parser.__class__.__name__ == "PDFParser"

    @patch("src.core.pipeline.create_storage")
    def test_get_parser_unknown_extension_raises(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No parser found"):
            pipeline._get_parser("file.xyz")

    @patch("src.core.pipeline.create_storage")
    def test_get_parser_case_insensitive(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        pipeline = Pipeline()
        parser = pipeline._get_parser("FILE.PDF")
        assert parser.__class__.__name__ == "PDFParser"

    @patch("src.core.pipeline.create_storage")
    def test_get_parser_no_extension_raises(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No parser found"):
            pipeline._get_parser("noextension")

    @patch("src.core.pipeline.create_storage")
    async def test_run_success(self, mock_create_storage):
        mock_storage = AsyncMock()
        mock_storage.download_bytes.return_value = b"pdf-data"
        mock_create_storage.return_value = mock_storage

        pipeline = Pipeline()

        mock_parsed = MagicMock()
        mock_parsed.content = "parsed content"
        mock_parsed.metadata = {"filename": "test.pdf"}

        mock_chunks = [MagicMock(content="chunk1", index=0, metadata={})]

        mock_embedded = [
            EmbeddedChunk(content="chunk1", embedding=[0.1, 0.2], index=0, metadata={})
        ]

        with (
            patch.object(pipeline, "_parse", return_value=mock_parsed),
            patch.object(pipeline, "_chunk", return_value=mock_chunks),
            patch.object(
                pipeline, "_embed", new_callable=AsyncMock, return_value=mock_embedded
            ),
        ):
            result = await pipeline.run(
                job_id="job-1",
                files=[
                    {
                        "filename": "test.pdf",
                        "storage_path": "job-1/test.pdf",
                    }
                ],
            )

        assert isinstance(result, PipelineResult)
        assert result.job_id == "job-1"
        assert result.total_chunks == 1
        assert len(result.embedded_chunks) == 1
        assert result.errors == []

    @patch("src.core.pipeline.create_storage")
    async def test_run_handles_error_per_file(self, mock_create_storage):
        mock_storage = AsyncMock()
        mock_storage.download_bytes.side_effect = Exception("download failed")
        mock_create_storage.return_value = mock_storage

        pipeline = Pipeline()

        result = await pipeline.run(
            job_id="job-2",
            files=[
                {
                    "filename": "bad.pdf",
                    "storage_path": "job-2/bad.pdf",
                }
            ],
        )

        assert len(result.errors) == 1
        assert "bad.pdf" in result.errors[0]
        assert "download failed" in result.errors[0]

    @patch("src.core.pipeline.create_storage")
    async def test_run_multiple_files(self, mock_create_storage):
        mock_storage = AsyncMock()
        mock_storage.download_bytes.return_value = b"data"
        mock_create_storage.return_value = mock_storage

        pipeline = Pipeline()

        mock_embedded = [
            EmbeddedChunk(content="c", embedding=[0.1], index=0, metadata={})
        ]

        with (
            patch.object(
                pipeline,
                "_parse",
                return_value=MagicMock(content="text", metadata={}),
            ),
            patch.object(
                pipeline,
                "_chunk",
                return_value=[MagicMock(content="c", index=0, metadata={})],
            ),
            patch.object(
                pipeline,
                "_embed",
                new_callable=AsyncMock,
                return_value=mock_embedded,
            ),
        ):
            result = await pipeline.run(
                job_id="job-3",
                files=[
                    {"filename": "a.pdf", "storage_path": "job-3/a.pdf"},
                    {"filename": "b.pdf", "storage_path": "job-3/b.pdf"},
                ],
            )

        assert result.total_chunks == 2
        assert len(result.embedded_chunks) == 2

    @patch("src.core.pipeline.create_storage")
    async def test_run_continues_after_file_error(self, mock_create_storage):
        call_count = 0

        async def download_side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("first file failed")
            return b"data"

        mock_storage = AsyncMock()
        mock_storage.download_bytes.side_effect = download_side_effect
        mock_create_storage.return_value = mock_storage

        pipeline = Pipeline()

        mock_embedded = [
            EmbeddedChunk(content="c", embedding=[0.1], index=0, metadata={})
        ]

        with (
            patch.object(
                pipeline,
                "_parse",
                return_value=MagicMock(content="text", metadata={}),
            ),
            patch.object(
                pipeline,
                "_chunk",
                return_value=[MagicMock(content="c", index=0, metadata={})],
            ),
            patch.object(
                pipeline,
                "_embed",
                new_callable=AsyncMock,
                return_value=mock_embedded,
            ),
        ):
            result = await pipeline.run(
                job_id="job-4",
                files=[
                    {"filename": "fail.pdf", "storage_path": "job-4/fail.pdf"},
                    {"filename": "ok.pdf", "storage_path": "job-4/ok.pdf"},
                ],
            )

        assert len(result.errors) == 1
        assert result.total_chunks == 1
        assert len(result.embedded_chunks) == 1

    @patch("src.core.pipeline.create_storage")
    async def test_run_empty_files(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        pipeline = Pipeline()

        result = await pipeline.run(job_id="job-5", files=[])

        assert result.total_chunks == 0
        assert result.embedded_chunks == []
        assert result.errors == []

    @patch("src.core.pipeline.create_storage")
    async def test_pipeline_result_dataclass(self, mock_create_storage):
        mock_create_storage.return_value = MagicMock()
        result = PipelineResult(job_id="test")
        assert result.job_id == "test"
        assert result.total_chunks == 0
        assert result.embedded_chunks == []
        assert result.errors == []
