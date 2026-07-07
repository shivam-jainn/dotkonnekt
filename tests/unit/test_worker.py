import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.pipeline import PipelineResult
from src.models.job import JobStatus
from src.worker.worker import Worker


@pytest.mark.unit
class TestWorker:
    @patch("src.core.tracing.collector.StructuredTraceCollector")
    @patch("src.core.agent.graph.create_graph")
    @patch("src.worker.worker.db")
    async def test_process_job_updates_status_to_completed(
        self, mock_db, mock_create_graph, mock_tracer_cls
    ):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "chunks": [1, 2, 3, 4, 5],
            "errors": []
        }
        mock_create_graph.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer
        
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_session.execute = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()
        # Mock _publish_for_storage since it's a separate integration concern
        worker._publish_for_storage = AsyncMock()
        
        job_data = {
            "job_id": "job-1",
            "status": "queued",
            "files": [
                {
                    "filename": "test.pdf",
                    "content_type": "application/pdf",
                    "size": 100,
                    "storage_path": "job-1/test.pdf",
                }
            ],
        }

        await worker._process_job(json.dumps(job_data).encode())

        assert mock_job.status == JobStatus.completed.value
        mock_tracer.flush_to_db.assert_awaited_once()
        mock_graph.ainvoke.assert_awaited_once()

    @patch("src.core.tracing.collector.StructuredTraceCollector")
    @patch("src.core.agent.graph.create_graph")
    @patch("src.worker.worker.db")
    async def test_process_job_updates_status_to_processing(
        self, mock_db, mock_create_graph, mock_tracer_cls
    ):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "chunks": [],
            "errors": []
        }
        mock_create_graph.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer
        
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_session.execute = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()
        worker._publish_for_storage = AsyncMock()
        
        job_data = {
            "job_id": "job-1",
            "status": "queued",
            "files": [],
        }

        await worker._process_job(json.dumps(job_data).encode())

        assert mock_session.get.await_count == 2
        assert mock_session.commit.await_count >= 2

    @patch("src.core.tracing.collector.StructuredTraceCollector")
    @patch("src.core.agent.graph.create_graph")
    @patch("src.worker.worker.db")
    async def test_process_job_calls_pipeline_run(self, mock_db, mock_create_graph, mock_tracer_cls):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "chunks": [],
            "errors": []
        }
        mock_create_graph.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_session.execute = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()
        worker._publish_for_storage = AsyncMock()
        
        job_data = {
            "job_id": "job-1",
            "status": "queued",
            "files": [
                {
                    "filename": "a.pdf",
                    "content_type": "application/pdf",
                    "size": 10,
                    "storage_path": "job-1/a.pdf",
                }
            ],
        }

        await worker._process_job(json.dumps(job_data).encode())

        mock_graph.ainvoke.assert_awaited_once()
        call_args, call_kwargs = mock_graph.ainvoke.call_args
        assert call_args[0]["job_id"] == "job-1"
        assert len(call_args[0]["files"]) == 1

    @patch("src.core.tracing.collector.StructuredTraceCollector")
    @patch("src.core.agent.graph.create_graph")
    @patch("src.worker.worker.db")
    async def test_process_job_updates_to_failed_on_exception(
        self, mock_db, mock_create_graph, mock_tracer_cls
    ):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = Exception("graph crashed")
        mock_create_graph.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()
        job_data = {
            "job_id": "job-1",
            "status": "queued",
            "files": [],
        }

        with pytest.raises(Exception):
            await worker._process_job(json.dumps(job_data).encode())

        assert mock_job.status == JobStatus.failed.value

    @patch("src.worker.worker.Pipeline")
    @patch("src.worker.worker.db")
    async def test_process_job_invalid_json_handles_gracefully(
        self, mock_db, mock_pipeline_cls
    ):
        mock_pipeline_cls.return_value = AsyncMock()

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()

        with pytest.raises(Exception):
            await worker._process_job(b"not-valid-json")

        mock_session.get.assert_not_called()

    @patch("asyncio.Event.wait")
    @patch("src.worker.worker.queue")
    @patch("src.worker.worker.settings")
    async def test_start_calls_consume(self, mock_settings, mock_queue, mock_wait):
        mock_settings.rabbitmq_queue = "ingestion"
        mock_queue.consume = AsyncMock()
        mock_wait = AsyncMock()

        worker = Worker()
        await worker.start()

        mock_queue.consume.assert_awaited_once()
        assert worker._running is True

    @patch("src.worker.worker.queue")
    @patch("src.worker.worker.settings")
    async def test_start_is_idempotent(self, mock_settings, mock_queue):
        mock_settings.rabbitmq_queue = "ingestion"
        mock_queue.consume = AsyncMock()

        worker = Worker()
        worker._running = True
        await worker.start()

        mock_queue.consume.assert_not_called()

    async def test_stop_sets_running_false(self):
        worker = Worker()
        worker._running = True
        await worker.stop()
        assert worker._running is False

    @patch("src.core.tracing.collector.StructuredTraceCollector")
    @patch("src.core.agent.graph.create_graph")
    @patch("src.worker.worker.db")
    async def test_process_job_passes_files_to_pipeline(
        self, mock_db, mock_create_graph, mock_tracer_cls
    ):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "chunks": [],
            "errors": []
        }
        mock_create_graph.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_session = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "j1"
        mock_session.get = AsyncMock(return_value=mock_job)
        mock_session.execute = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db.pool.return_value = mock_cm

        worker = Worker()
        worker._publish_for_storage = AsyncMock()
        
        job_data = {
            "job_id": "j1",
            "status": "queued",
            "files": [
                {
                    "filename": "a.pdf",
                    "content_type": "application/pdf",
                    "size": 1,
                    "storage_path": "j1/a.pdf",
                },
                {
                    "filename": "b.pdf",
                    "content_type": "application/pdf",
                    "size": 2,
                    "storage_path": "j1/b.pdf",
                },
            ],
        }

        await worker._process_job(json.dumps(job_data).encode())

        call_args, call_kwargs = mock_graph.ainvoke.call_args
        assert len(call_args[0]["files"]) == 2
        assert call_args[0]["files"][0]["filename"] == "a.pdf"
        assert call_args[0]["files"][1]["filename"] == "b.pdf"
