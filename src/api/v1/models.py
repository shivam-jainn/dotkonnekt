from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from src.core.models.providers import ProviderConfig, TaskType
from src.core.models.registry import registry
from src.core.models.schemas import ConnectProviderRequest, UpdateModelConfigRequest

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/providers")
async def list_providers(
    task: Annotated[TaskType | None, Query(description="Filter by task type")] = None,
):
    return registry.list_providers(task)


@router.post(
    "/providers/{provider_id}/connect",
    status_code=status.HTTP_200_OK,
)
async def connect_provider(provider_id: str, body: ConnectProviderRequest):
    try:
        config = ProviderConfig(
            api_key=body.api_key,
            api_base=body.api_base,
            extra_headers=body.extra_headers,
            extra_params=body.extra_params,
        )
        result = await registry.connect_provider(provider_id, config)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to provider: {e}",
        )


@router.get("/providers/{provider_id}/models")
async def list_models(provider_id: str):
    models = registry.list_models(provider_id)
    if not models:
        status_ = registry.get_provider_status(provider_id)
        if status_.value == "disconnected":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider_id}' is not connected. Connect first.",
            )
    return models


@router.get("/config")
async def get_model_config():
    selected = registry.get_selected_config()
    return {
        "embedding": selected.get(TaskType.EMBEDDING),
        "llm": selected.get(TaskType.LLM),
        "reranker": selected.get(TaskType.RERANKER),
    }


@router.put("/config")
async def update_model_config(body: UpdateModelConfigRequest):
    try:
        entry = registry.select_model(body.task, body.provider_id, body.model_id)
        return entry
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
