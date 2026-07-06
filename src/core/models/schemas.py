from typing import Any

from pydantic import BaseModel

from src.core.models.providers import ProviderStatus, TaskType


class ProviderInfo(BaseModel):
    id: str
    name: str
    supported_tasks: list[TaskType]
    requires_api_key: bool
    default_api_base: str | None = None
    description: str
    status: ProviderStatus = ProviderStatus.DISCONNECTED
    connected_models: list[str] = []


class ConnectProviderRequest(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] = {}
    extra_params: dict[str, Any] = {}


class ModelInfo(BaseModel):
    id: str
    litellm_model: str
    owned_by: str | None = None
    task: TaskType


class ModelConfigEntry(BaseModel):
    task: TaskType
    provider_id: str
    model_id: str
    litellm_model: str
    api_base: str | None = None


class ModelConfigResponse(BaseModel):
    embedding: ModelConfigEntry | None = None
    llm: ModelConfigEntry | None = None
    reranker: ModelConfigEntry | None = None


class UpdateModelConfigRequest(BaseModel):
    task: TaskType
    provider_id: str
    model_id: str
