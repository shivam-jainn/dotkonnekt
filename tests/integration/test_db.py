from uuid import uuid4

import pytest
from sqlalchemy import text

from src.database import db
from src.database.models import JobModel


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def clean_jobs():
    await db.connect()
    async with db.pool() as session:
        await session.execute(text("DELETE FROM jobs"))
        await session.commit()
    yield
    async with db.pool() as session:
        await session.execute(text("DELETE FROM jobs"))
        await session.commit()
    await db.close()


class TestDatabaseIntegration:
    async def test_schema_exists(self):
        async with db.pool() as session:
            result = await session.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_name = 'jobs'")
            )
            rows = result.fetchall()
            assert len(rows) == 1

    async def test_insert_and_read_job(self):
        job_id = str(uuid4())
        files = [
            {"filename": "test.txt", "content_type": "text/plain", "size": 12, "storage_path": f"{job_id}/test.txt"}
        ]
        collection = "test-collection"
        metadata = {"source": "pytest"}

        async with db.pool() as session:
            session.add(JobModel(
                id=job_id,
                status="queued",
                files=files,
                collection=collection,
                metadata_=metadata,
            ))
            await session.commit()

        async with db.pool() as session:
            result = await session.execute(
                text("SELECT * FROM jobs WHERE id = :id"), {"id": job_id}
            )
            row = result.fetchone()

        assert row is not None
        assert row._mapping["id"] == job_id
        assert row._mapping["status"] == "queued"
        assert row._mapping["files"] == files
        assert row._mapping["collection"] == collection
        assert row._mapping["metadata"] == metadata
        assert row._mapping["created_at"] is not None
        assert row._mapping["updated_at"] is not None

    async def test_default_status_is_queued(self):
        job_id = str(uuid4())

        async with db.pool() as session:
            session.add(JobModel(id=job_id))
            await session.commit()

        async with db.pool() as session:
            result = await session.execute(
                text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id}
            )
            row = result.fetchone()

        assert row._mapping["status"] == "queued"

    async def test_update_job_status(self):
        job_id = str(uuid4())

        async with db.pool() as session:
            session.add(JobModel(id=job_id))
            await session.commit()

        async with db.pool() as session:
            await session.execute(
                text("UPDATE jobs SET status = :status, updated_at = NOW() WHERE id = :id"),
                {"status": "processing", "id": job_id},
            )
            await session.commit()

        async with db.pool() as session:
            result = await session.execute(
                text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id}
            )
            row = result.fetchone()

        assert row._mapping["status"] == "processing"

    async def test_multiple_files_jsonb(self):
        job_id = str(uuid4())
        files = [
            {"filename": "a.txt", "content_type": "text/plain", "size": 1, "storage_path": f"{job_id}/a.txt"},
            {"filename": "b.txt", "content_type": "text/plain", "size": 2, "storage_path": f"{job_id}/b.txt"},
        ]

        async with db.pool() as session:
            session.add(JobModel(id=job_id, files=files))
            await session.commit()

        async with db.pool() as session:
            result = await session.execute(
                text("SELECT files FROM jobs WHERE id = :id"), {"id": job_id}
            )
            row = result.fetchone()

        assert len(row._mapping["files"]) == 2
