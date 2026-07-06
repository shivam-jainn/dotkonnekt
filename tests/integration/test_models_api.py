from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.router import api_router
from src.core.models.providers import TaskType
from src.core.models.registry import registry


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_registry():
    orig_selected = dict(registry._selected)
    orig_models = dict(registry._models)
    orig_configs = dict(registry._configs)
    orig_status = dict(registry._status)
    registry._selected = {}
    registry._models = {}
    registry._configs = {}
    registry._status = {}
    yield
    registry._selected = orig_selected
    registry._models = orig_models
    registry._configs = orig_configs
    registry._status = orig_status


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


class TestModelsApi:
    async def test_list_providers(self, client):
        response = await client.get("/api/v1/models/providers")

        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        provider_ids = [p["id"] for p in data]
        assert "openai" in provider_ids
        assert "lmstudio" in provider_ids
        assert "ollama" in provider_ids

    async def test_list_providers_filter_embedding(self, client):
        response = await client.get("/api/v1/models/providers?task=embedding")

        assert response.status_code == 200
        data = response.json()
        for p in data:
            assert "embedding" in p["supported_tasks"]

    async def test_list_providers_filter_llm(self, client):
        response = await client.get("/api/v1/models/providers?task=llm")

        assert response.status_code == 200
        data = response.json()
        for p in data:
            assert "llm" in p["supported_tasks"]

    async def test_connect_provider_unknown(self, client):
        response = await client.post(
            "/api/v1/models/providers/nonexistent/connect",
            json={},
        )

        assert response.status_code == 400

    @patch("src.core.models.registry.httpx")
    async def test_connect_lmstudio(self, mock_httpx, client):
        from src.core.models.registry import registry

        original_fetch = registry._fetch_lmstudio_models

        async def fake_fetch(base_url):
            from src.core.models.schemas import ModelInfo
            from src.core.models.providers import TaskType

            return [
                ModelInfo(
                    id="nomic-embed",
                    litellm_model="openai/nomic-embed",
                    task=TaskType.EMBEDDING,
                ),
            ]

        registry._fetch_lmstudio_models = fake_fetch
        try:
            response = await client.post(
                "/api/v1/models/providers/lmstudio/connect",
                json={"api_base": "http://localhost:1234/v1"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["provider_id"] == "lmstudio"
            assert data["status"] == "connected"
        finally:
            registry._fetch_lmstudio_models = original_fetch

    async def test_list_models_not_connected(self, client):
        response = await client.get("/api/v1/models/providers/openai/models")

        assert response.status_code == 400

    async def test_get_config_empty(self, client):
        response = await client.get("/api/v1/models/config")

        assert response.status_code == 200
        data = response.json()
        assert data["embedding"] is None
        assert data["llm"] is None

    async def test_update_config_unknown_provider(self, client):
        response = await client.put(
            "/api/v1/models/config",
            json={
                "task": "embedding",
                "provider_id": "nonexistent",
                "model_id": "model",
            },
        )

        assert response.status_code == 400

    @patch("src.core.models.registry.httpx")
    async def test_full_flow_connect_select_get_config(self, mock_httpx, client):
        from src.core.models.registry import registry

        mock_models = [
            {
                "id": "nomic-embed-text",
                "litellm_model": "openai/nomic-embed-text",
                "task": "embedding",
            },
            {"id": "llama-3", "litellm_model": "openai/llama-3", "task": "llm"},
        ]

        original_fetch = registry._fetch_lmstudio_models

        async def fake_fetch(base_url):
            from src.core.models.schemas import ModelInfo
            from src.core.models.providers import TaskType

            return [
                ModelInfo(
                    id=m["id"],
                    litellm_model=m["litellm_model"],
                    task=TaskType(m["task"]),
                )
                for m in mock_models
            ]

        registry._fetch_lmstudio_models = fake_fetch
        try:
            await client.post(
                "/api/v1/models/providers/lmstudio/connect",
                json={"api_base": "http://localhost:1234/v1"},
            )

            models_resp = await client.get("/api/v1/models/providers/lmstudio/models")
            assert models_resp.status_code == 200
            models = models_resp.json()
            assert len(models) > 0

            select_resp = await client.put(
                "/api/v1/models/config",
                json={
                    "task": "embedding",
                    "provider_id": "lmstudio",
                    "model_id": "nomic-embed-text",
                },
            )
            assert select_resp.status_code == 200

            config_resp = await client.get("/api/v1/models/config")
            assert config_resp.status_code == 200
            config = config_resp.json()
            assert config["embedding"]["provider_id"] == "lmstudio"
            assert config["embedding"]["model_id"] == "nomic-embed-text"
        finally:
            registry._fetch_lmstudio_models = original_fetch
