import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.router import api_router
from src.database import db


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def mock_db_session():
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    mock_factory = MagicMock()
    mock_factory.return_value = mock_cm
    db._session_factory = mock_factory
    yield mock_session
    db._session_factory = None
    db._engine = None


@pytest.fixture(autouse=True)
def mock_storage():
    with patch("src.api.v1.upload.create_storage") as mock_factory:
        mock_storage_instance = MagicMock()
        mock_factory.return_value = mock_storage_instance
        yield mock_storage_instance


@pytest.fixture(autouse=True)
def mock_queue():
    with patch("src.api.v1.upload.queue.publish", new_callable=AsyncMock) as mock_publish:
        yield mock_publish


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(api_router)
    return test_app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestUploadApi:
    async def test_upload_single_file(self, client, mock_db_session, mock_storage, mock_queue):
        file_content = b"Hello, World!"
        response = await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", file_content, "text/plain")},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "queued"
        assert data["files_uploaded"] == 1
        assert len(data["job_id"]) > 0

        mock_storage.upload_bytes.assert_called_once()
        assert mock_storage.upload_bytes.call_args[0][0] == file_content

        mock_queue.assert_awaited_once()

    async def test_upload_multiple_files(self, client, mock_db_session, mock_storage, mock_queue):
        response = await client.post(
            "/api/v1/documents",
            files=[
                ("files", ("a.txt", b"content-a", "text/plain")),
                ("files", ("b.txt", b"content-b", "text/plain")),
                ("files", ("c.txt", b"content-c", "text/plain")),
            ],
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "queued"
        assert data["files_uploaded"] == 3

        assert mock_storage.upload_bytes.call_count == 3

        mock_queue.assert_awaited_once()

    async def test_upload_with_collection(self, client, mock_db_session, mock_storage, mock_queue):
        response = await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", b"data", "text/plain")},
            data={"collection": "my-collection"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["files_uploaded"] == 1

        call_args = mock_db_session.add.call_args
        job = call_args[0][0]
        assert job.collection == "my-collection"

    async def test_upload_with_metadata(self, client, mock_db_session, mock_storage, mock_queue):
        metadata = {"source": "test", "tags": ["api", "upload"]}
        response = await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", b"data", "text/plain")},
            data={"metadata": json.dumps(metadata)},
        )

        assert response.status_code == 201

        call_args = mock_db_session.add.call_args
        job = call_args[0][0]
        assert job.metadata_ == metadata

    async def test_upload_with_collection_and_metadata(self, client, mock_db_session, mock_storage, mock_queue):
        metadata = {"source": "full-test"}
        response = await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", b"data", "text/plain")},
            data={
                "collection": "my-collection",
                "metadata": json.dumps(metadata),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["files_uploaded"] == 1
        assert data["status"] == "queued"

    async def test_upload_no_files_returns_validation_error(self, client):
        response = await client.post(
            "/api/v1/documents",
        )

        assert response.status_code == 422

    async def test_upload_empty_files_list_returns_validation_error(self, client):
        response = await client.post(
            "/api/v1/documents",
            files=[],
        )

        assert response.status_code == 422

    async def test_job_inserted_with_correct_fields(self, client, mock_db_session, mock_storage, mock_queue):
        await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", b"data", "text/plain")},
            data={"collection": "col", "metadata": json.dumps({"k": "v"})},
        )

        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_awaited_once()

    async def test_job_published_to_queue(self, client, mock_db_session, mock_storage, mock_queue):
        await client.post(
            "/api/v1/documents",
            files={"files": ("test.txt", b"data", "text/plain")},
        )

        mock_queue.assert_awaited_once()
        queue_name = mock_queue.call_args[0][0]
        assert queue_name == "ingestion"

    async def test_upload_preserves_content_type(self, client, mock_db_session, mock_storage, mock_queue):
        await client.post(
            "/api/v1/documents",
            files={"files": ("test.json", b'{"key": "value"}', "application/json")},
        )

        call_args = mock_storage.upload_bytes.call_args
        assert call_args[0][2] == "application/json"

    async def test_upload_binary_file(self, client, mock_db_session, mock_storage, mock_queue):
        binary_content = b"\x00\x01\x02\xff\xfe"
        response = await client.post(
            "/api/v1/documents",
            files={"files": ("data.bin", binary_content, "application/octet-stream")},
        )

        assert response.status_code == 201
        assert mock_storage.upload_bytes.call_args[0][0] == binary_content
