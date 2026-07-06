import json
from uuid import uuid4

import pytest

from src.database import db


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def clean_jobs():
    await db.connect()
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM jobs")
    yield
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM jobs")
    await db.close()


class TestDatabaseIntegration:
    async def test_schema_auto_created(self):
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'jobs'"
            )
            assert len(rows) == 1

    async def test_insert_and_read_job(self):
        job_id = str(uuid4())
        files = [
            {"filename": "test.txt", "content_type": "text/plain", "size": 12, "storage_path": f"{job_id}/test.txt"}
        ]
        collection = "test-collection"
        metadata = {"source": "pytest"}

        async with db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (id, status, files, collection, metadata)
                VALUES ($1, $2, $3::jsonb, $4, $5::jsonb)
                """,
                job_id,
                "queued",
                json.dumps(files),
                collection,
                json.dumps(metadata),
            )

        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)

        assert row is not None
        assert row["id"] == job_id
        assert row["status"] == "queued"
        assert json.loads(row["files"]) == files
        assert row["collection"] == collection
        assert json.loads(row["metadata"]) == metadata
        assert row["created_at"] is not None
        assert row["updated_at"] is not None

    async def test_default_status_is_queued(self):
        job_id = str(uuid4())

        async with db.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO jobs (id) VALUES ($1)", job_id
            )

        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT status FROM jobs WHERE id = $1", job_id)

        assert row["status"] == "queued"

    async def test_update_job_status(self):
        job_id = str(uuid4())

        async with db.pool.acquire() as conn:
            await conn.execute("INSERT INTO jobs (id) VALUES ($1)", job_id)
            await conn.execute(
                "UPDATE jobs SET status = $1, updated_at = NOW() WHERE id = $2",
                "processing",
                job_id,
            )

        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT status FROM jobs WHERE id = $1", job_id)

        assert row["status"] == "processing"

    async def test_multiple_files_jsonb(self):
        job_id = str(uuid4())
        files = [
            {"filename": "a.txt", "content_type": "text/plain", "size": 1, "storage_path": f"{job_id}/a.txt"},
            {"filename": "b.txt", "content_type": "text/plain", "size": 2, "storage_path": f"{job_id}/b.txt"},
        ]

        async with db.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO jobs (id, files) VALUES ($1, $2::jsonb)",
                job_id,
                json.dumps(files),
            )

        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT files FROM jobs WHERE id = $1", job_id)

        assert len(json.loads(row["files"])) == 2
