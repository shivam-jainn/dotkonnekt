import json
import logging
from pathlib import Path

import httpx

from src.core.models.providers import (
    PROVIDERS,
    ProviderConfig,
    ProviderMeta,
    TaskType,
    get_provider,
)
from src.core.models.schemas import ModelConfigEntry, ModelInfo, ProviderStatus

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("model_config.json")


class ModelRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, ProviderConfig] = {}
        self._status: dict[str, ProviderStatus] = {}
        self._models: dict[str, list[ModelInfo]] = {}
        self._selected: dict[TaskType, ModelConfigEntry] = {}
        self._load_config()

    def _load_config(self) -> None:
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                for pid, pcfg in data.get("providers", {}).items():
                    self._configs[pid] = ProviderConfig(**pcfg)
                    self._status[pid] = ProviderStatus.CONNECTED
                    meta = get_provider(pid)
                    if meta:
                        self._models[pid] = self._static_models(meta)
                for task_str, entry in data.get("selected", {}).items():
                    self._selected[TaskType(task_str)] = ModelConfigEntry(**entry)
                logger.info("Loaded model config from %s", CONFIG_PATH)
            except Exception:
                logger.exception("Failed to load model config")

    def _save_config(self) -> None:
        data = {
            "providers": {
                pid: {
                    "api_key": cfg.api_key,
                    "api_base": cfg.api_base,
                    "extra_headers": cfg.extra_headers,
                    "extra_params": cfg.extra_params,
                }
                for pid, cfg in self._configs.items()
            },
            "selected": {
                task.value: entry.model_dump() for task, entry in self._selected.items()
            },
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))

    def list_providers(self, task: TaskType | None = None) -> list[dict]:
        results = []
        for meta in PROVIDERS:
            if task and task not in meta.supported_tasks:
                continue
            status = self._status.get(meta.id, ProviderStatus.DISCONNECTED)
            connected_models = [m.id for m in self._models.get(meta.id, [])]
            results.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "supported_tasks": meta.supported_tasks,
                    "requires_api_key": meta.requires_api_key,
                    "default_api_base": meta.default_api_base,
                    "description": meta.description,
                    "status": status,
                    "connected_models": connected_models,
                }
            )
        return results

    async def connect_provider(self, provider_id: str, config: ProviderConfig) -> dict:
        meta = get_provider(provider_id)
        if meta is None:
            raise ValueError(f"Unknown provider: {provider_id}")

        api_base = config.api_base or meta.default_api_base

        self._configs[provider_id] = config
        self._status[provider_id] = ProviderStatus.CONNECTED

        models = await self._fetch_models(provider_id, meta, api_base, config)
        self._models[provider_id] = models

        self._save_config()

        return {
            "provider_id": provider_id,
            "status": "connected",
            "models_found": len(models),
        }

    async def _fetch_models(
        self,
        provider_id: str,
        meta: ProviderMeta,
        api_base: str | None,
        config: ProviderConfig,
    ) -> list[ModelInfo]:
        if provider_id == "lmstudio":
            return await self._fetch_lmstudio_models(
                api_base or "http://localhost:1234/v1"
            )

        if provider_id == "ollama":
            return await self._fetch_ollama_models(api_base or "http://localhost:11434")

        if api_base and provider_id not in ("anthropic", "bedrock", "vertex_ai"):
            return await self._fetch_openai_compatible_models(api_base, config)

        return self._static_models(meta)

    async def _fetch_lmstudio_models(self, base_url: str) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/models")
                resp.raise_for_status()
                data = resp.json()

            models = []
            for m in data.get("data", []):
                model_id = m["id"]
                models.append(
                    ModelInfo(
                        id=model_id,
                        litellm_model=f"openai/{model_id}",
                        owned_by=m.get("owned_by", "lmstudio"),
                        task=TaskType.LLM,
                    )
                )

            embedding_models = [
                m
                for m in models
                if any(
                    kw in m.id.lower()
                    for kw in ("embed", "bge", "e5", "nomic", "gte", "minilm")
                )
            ]
            for m in embedding_models:
                m.task = TaskType.EMBEDDING

            if not embedding_models:
                for m in models:
                    models.append(
                        ModelInfo(
                            id=m.id,
                            litellm_model=f"openai/{m.id}",
                            owned_by="lmstudio",
                            task=TaskType.EMBEDDING,
                        )
                    )

            return models
        except Exception:
            logger.warning("Failed to fetch LM Studio models from %s", base_url)
            return []

    async def _fetch_ollama_models(self, base_url: str) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

            models = []
            for m in data.get("models", []):
                model_id = m["name"]
                models.append(
                    ModelInfo(
                        id=model_id,
                        litellm_model=f"ollama/{model_id}",
                        owned_by="ollama",
                        task=TaskType.LLM,
                    )
                )
            return models
        except Exception:
            logger.warning("Failed to fetch Ollama models from %s", base_url)
            return []

    async def _fetch_openai_compatible_models(
        self, base_url: str, config: ProviderConfig
    ) -> list[ModelInfo]:
        try:
            headers = (
                {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
            )
            url = base_url.rstrip("/")
            if not url.endswith("/v1"):
                url += "/v1"
            url += "/models"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            models = []
            for m in data.get("data", []):
                model_id = m["id"]
                task = TaskType.LLM
                if any(kw in model_id.lower() for kw in ("embed", "bge", "e5")):
                    task = TaskType.EMBEDDING
                models.append(
                    ModelInfo(
                        id=model_id,
                        litellm_model=model_id,
                        owned_by=m.get("owned_by"),
                        task=task,
                    )
                )
            return models
        except Exception:
            logger.warning("Failed to fetch models from %s", base_url)
            return []

    def _static_models(self, meta: ProviderMeta) -> list[ModelInfo]:
        import litellm

        provider_key = meta.litellm_prefix
        if provider_key not in litellm.models_by_provider:
            provider_key = meta.id

        litellm_models = list(litellm.models_by_provider.get(provider_key, []))

        if not litellm_models:
            try:
                litellm_models = list(litellm.get_valid_models(custom_llm_provider=meta.litellm_prefix))
            except Exception:
                pass
        if not litellm_models:
            try:
                litellm_models = list(litellm.get_valid_models(custom_llm_provider=meta.id))
            except Exception:
                pass

        models = []
        registered_ids = set()

        def add_model(model_id: str, litellm_model: str, task: TaskType):
            if model_id not in registered_ids:
                models.append(
                    ModelInfo(
                        id=model_id,
                        litellm_model=litellm_model,
                        task=task,
                    )
                )
                registered_ids.add(model_id)

        for model_name in litellm_models:
            task = TaskType.LLM
            model_name_lower = model_name.lower()
            if any(
                kw in model_name_lower
                for kw in ("embed", "bge", "e5", "nomic", "gte", "minilm", "similarity")
            ):
                task = TaskType.EMBEDDING

            if task not in meta.supported_tasks:
                continue

            # Add original model name
            add_model(model_name, model_name, task)

            # Generate stripped alias by removing provider prefix if present (e.g. groq/qwen/qwen3-32b -> qwen/qwen3-32b)
            if model_name.startswith(f"{meta.litellm_prefix}/"):
                stripped = model_name[len(meta.litellm_prefix) + 1:]
                add_model(stripped, model_name, task)

            # Generate other aliases
            parts = model_name.split("/")
            if len(parts) > 1:
                base = parts[-1]
                add_model(base, model_name, task)
                add_model(f"{meta.litellm_prefix}/{base}", model_name, task)
            else:
                # If there's no prefix in litellm model name, add prefix/name as an option
                add_model(f"{meta.litellm_prefix}/{model_name}", model_name, task)

        return models

    def get_provider_status(self, provider_id: str) -> ProviderStatus:
        return self._status.get(provider_id, ProviderStatus.DISCONNECTED)

    def get_provider_config(self, provider_id: str) -> ProviderConfig | None:
        return self._configs.get(provider_id)

    def list_models(self, provider_id: str) -> list[ModelInfo]:
        return self._models.get(provider_id, [])

    def get_selected_config(self) -> dict[TaskType, ModelConfigEntry | None]:
        return dict(self._selected)

    def select_model(
        self, task: TaskType, provider_id: str, model_id: str
    ) -> ModelConfigEntry:
        meta = get_provider(provider_id)
        if meta is None:
            raise ValueError(f"Unknown provider: {provider_id}")

        models = self._models.get(provider_id, [])
        model = next((m for m in models if m.id == model_id), None)
        if model is None:
            raise ValueError(
                f"Model '{model_id}' not found for provider '{provider_id}'. "
                f"Available: {[m.id for m in models]}"
            )

        cfg = self._configs.get(provider_id)
        api_base = None
        if cfg and cfg.api_base:
            api_base = cfg.api_base
        elif meta.default_api_base:
            api_base = meta.default_api_base

        entry = ModelConfigEntry(
            task=task,
            provider_id=provider_id,
            model_id=model_id,
            litellm_model=model.litellm_model,
            api_base=api_base,
        )

        self._selected[task] = entry
        self._save_config()
        return entry

    def get_litellm_kwargs(self, task: TaskType) -> dict:
        entry = self._selected.get(task)
        if entry is None:
            return {}

        kwargs: dict = {"model": entry.litellm_model}

        if entry.api_base:
            kwargs["api_base"] = entry.api_base

        cfg = self._configs.get(entry.provider_id)
        if cfg:
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.extra_headers:
                kwargs["extra_headers"] = cfg.extra_headers
            if cfg.extra_params:
                kwargs.update(cfg.extra_params)

        meta = get_provider(entry.provider_id)
        if meta:
            if meta.id != "openai":
                prefix = f"{meta.litellm_prefix}/"
                if not entry.litellm_model.startswith(prefix):
                    kwargs["model"] = f"{prefix}{entry.litellm_model}"
            if "api_key" not in kwargs and not meta.requires_api_key:
                kwargs["api_key"] = "not-needed"

        return kwargs


registry = ModelRegistry()
