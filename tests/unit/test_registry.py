import json
import os
from unittest.mock import AsyncMock

import pytest

from src.core.models.providers import ProviderConfig, TaskType
from src.core.models.registry import ModelRegistry
from src.core.models.schemas import ModelInfo


@pytest.mark.unit
class TestModelRegistry:
    def _make_registry(self, tmp_path):
        os.chdir(tmp_path)
        return ModelRegistry()

    def test_list_providers(self, tmp_path):
        reg = self._make_registry(tmp_path)
        providers = reg.list_providers()
        assert len(providers) > 0

    def test_list_providers_filter_task(self, tmp_path):
        reg = self._make_registry(tmp_path)
        embedding_providers = reg.list_providers(task=TaskType.EMBEDDING)
        for p in embedding_providers:
            assert TaskType.EMBEDDING in p["supported_tasks"]

    def test_get_provider_status_disconnected(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.get_provider_status("openai").value == "disconnected"

    def test_get_provider_config_none(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.get_provider_config("openai") is None

    def test_list_models_populated_for_disconnected_provider(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.get_provider_status("openai").value == "disconnected"
        assert len(reg.list_models("openai")) > 0

    def test_get_selected_config_empty(self, tmp_path):
        reg = self._make_registry(tmp_path)
        selected = reg.get_selected_config()
        assert selected.get(TaskType.EMBEDDING) is None
        assert selected.get(TaskType.LLM) is None

    def test_select_model_unknown_provider_raises(self, tmp_path):
        reg = self._make_registry(tmp_path)
        with pytest.raises(ValueError, match="Unknown provider"):
            reg.select_model(TaskType.EMBEDDING, "nonexistent", "model")

    def test_select_model_unknown_model_raises(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg._models["lmstudio"] = [
            ModelInfo(
                id="nomic-embed",
                litellm_model="openai/nomic-embed",
                task=TaskType.EMBEDDING,
            ),
        ]
        reg._configs["lmstudio"] = ProviderConfig(api_base="http://localhost:1234/v1")

        with pytest.raises(ValueError, match="not found"):
            reg.select_model(TaskType.EMBEDDING, "lmstudio", "nonexistent-model")

    def test_select_model_success(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg._models["lmstudio"] = [
            ModelInfo(
                id="nomic-embed",
                litellm_model="openai/nomic-embed",
                task=TaskType.EMBEDDING,
            ),
        ]
        reg._configs["lmstudio"] = ProviderConfig(api_base="http://localhost:1234/v1")

        entry = reg.select_model(TaskType.EMBEDDING, "lmstudio", "nomic-embed")

        assert entry.task == TaskType.EMBEDDING
        assert entry.provider_id == "lmstudio"
        assert entry.model_id == "nomic-embed"
        assert entry.litellm_model == "openai/nomic-embed"
        assert entry.api_base == "http://localhost:1234/v1"

    def test_select_model_persists_to_file(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg._models["lmstudio"] = [
            ModelInfo(
                id="nomic-embed",
                litellm_model="openai/nomic-embed",
                task=TaskType.EMBEDDING,
            ),
        ]
        reg._configs["lmstudio"] = ProviderConfig(api_base="http://localhost:1234/v1")

        reg.select_model(TaskType.EMBEDDING, "lmstudio", "nomic-embed")

        config_file = tmp_path / "model_config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "selected" in data
        assert "embedding" in data["selected"]

    def test_get_litellm_kwargs_no_selection(self, tmp_path):
        reg = self._make_registry(tmp_path)
        kwargs = reg.get_litellm_kwargs(TaskType.EMBEDDING)
        assert kwargs == {}

    def test_get_litellm_kwargs_with_selection(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg._models["lmstudio"] = [
            ModelInfo(
                id="nomic-embed",
                litellm_model="openai/nomic-embed",
                task=TaskType.EMBEDDING,
            ),
        ]
        reg._configs["lmstudio"] = ProviderConfig(api_base="http://localhost:1234/v1")
        reg.select_model(TaskType.EMBEDDING, "lmstudio", "nomic-embed")

        kwargs = reg.get_litellm_kwargs(TaskType.EMBEDDING)
        assert kwargs["model"] == "openai/nomic-embed"
        assert kwargs["api_base"] == "http://localhost:1234/v1"

    def test_get_litellm_kwargs_with_api_key(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg._models["openai"] = [
            ModelInfo(
                id="text-embedding-3-small",
                litellm_model="text-embedding-3-small",
                task=TaskType.EMBEDDING,
            ),
        ]
        reg._configs["openai"] = ProviderConfig(api_key="sk-test")
        reg.select_model(TaskType.EMBEDDING, "openai", "text-embedding-3-small")

        kwargs = reg.get_litellm_kwargs(TaskType.EMBEDDING)
        assert kwargs["api_key"] == "sk-test"

    def test_load_config_from_file(self, tmp_path):
        config_data = {
            "providers": {
                "lmstudio": {
                    "api_key": None,
                    "api_base": "http://localhost:1234/v1",
                    "extra_headers": {},
                    "extra_params": {},
                }
            },
            "selected": {
                "embedding": {
                    "task": "embedding",
                    "provider_id": "lmstudio",
                    "model_id": "nomic-embed",
                    "litellm_model": "openai/nomic-embed",
                    "api_base": "http://localhost:1234/v1",
                }
            },
        }
        (tmp_path / "model_config.json").write_text(json.dumps(config_data))

        reg = self._make_registry(tmp_path)

        assert reg.get_provider_status("lmstudio").value == "connected"
        cfg = reg.get_provider_config("lmstudio")
        assert cfg is not None
        assert cfg.api_base == "http://localhost:1234/v1"

    async def test_connect_lmstudio_fetches_models(self, tmp_path):
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            reg = ModelRegistry()

            mock_models = [
                ModelInfo(
                    id="nomic-embed-text",
                    litellm_model="openai/nomic-embed-text",
                    task=TaskType.EMBEDDING,
                ),
                ModelInfo(
                    id="llama-3", litellm_model="openai/llama-3", task=TaskType.LLM
                ),
            ]
            reg._fetch_lmstudio_models = AsyncMock(return_value=mock_models)

            config = ProviderConfig(api_base="http://localhost:1234/v1")
            result = await reg.connect_provider("lmstudio", config)

            assert result["provider_id"] == "lmstudio"
            assert result["status"] == "connected"
            assert result["models_found"] == 2

            models = reg.list_models("lmstudio")
            assert len(models) == 2
            assert models[0].litellm_model == "openai/nomic-embed-text"
        finally:
            os.chdir(orig_dir)
