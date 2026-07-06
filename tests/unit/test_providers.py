import pytest

from src.core.models.providers import (
    PROVIDERS,
    TaskType,
    get_provider,
    list_providers,
)


@pytest.mark.unit
class TestProviders:
    def test_list_all_providers(self):
        providers = list_providers()
        assert len(providers) > 0
        assert all(hasattr(p, "id") for p in providers)

    def test_list_providers_filter_by_embedding(self):
        providers = list_providers(task=TaskType.EMBEDDING)
        for p in providers:
            assert TaskType.EMBEDDING in p.supported_tasks

    def test_list_providers_filter_by_llm(self):
        providers = list_providers(task=TaskType.LLM)
        for p in providers:
            assert TaskType.LLM in p.supported_tasks

    def test_list_providers_filter_by_reranker(self):
        providers = list_providers(task=TaskType.RERANKER)
        for p in providers:
            assert TaskType.RERANKER in p.supported_tasks

    def test_get_provider_openai(self):
        p = get_provider("openai")
        assert p is not None
        assert p.name == "OpenAI"
        assert p.requires_api_key is True

    def test_get_provider_lmstudio(self):
        p = get_provider("lmstudio")
        assert p is not None
        assert p.name == "LM Studio"
        assert p.requires_api_key is False
        assert p.default_api_base == "http://localhost:1234/v1"

    def test_get_provider_ollama(self):
        p = get_provider("ollama")
        assert p is not None
        assert p.name == "Ollama"
        assert p.requires_api_key is False

    def test_get_provider_unknown(self):
        p = get_provider("nonexistent")
        assert p is None

    def test_lmstudio_supports_embedding_and_llm(self):
        p = get_provider("lmstudio")
        assert TaskType.EMBEDDING in p.supported_tasks
        assert TaskType.LLM in p.supported_tasks

    def test_anthropic_only_supports_llm(self):
        p = get_provider("anthropic")
        assert p is not None
        assert TaskType.LLM in p.supported_tasks
        assert TaskType.EMBEDDING not in p.supported_tasks

    def test_providers_have_litellm_prefix(self):
        for p in PROVIDERS:
            assert p.litellm_prefix
