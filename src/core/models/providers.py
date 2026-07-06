from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    EMBEDDING = "embedding"
    LLM = "llm"
    RERANKER = "reranker"


class ProviderStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ProviderConfig:
    api_key: str | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMeta:
    id: str
    name: str
    litellm_prefix: str
    supported_tasks: list[TaskType]
    requires_api_key: bool
    default_api_base: str | None = None
    description: str = ""


PROVIDERS: list[ProviderMeta] = [
    ProviderMeta(
        id="openai",
        name="OpenAI",
        litellm_prefix="openai",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM, TaskType.RERANKER],
        requires_api_key=True,
        description="OpenAI API (GPT, embeddings)",
    ),
    ProviderMeta(
        id="lmstudio",
        name="LM Studio",
        litellm_prefix="openai",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=False,
        default_api_base="http://localhost:1234/v1",
        description="LM Studio local server (OpenAI-compatible)",
    ),
    ProviderMeta(
        id="ollama",
        name="Ollama",
        litellm_prefix="ollama",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=False,
        default_api_base="http://localhost:11434",
        description="Ollama local models",
    ),
    ProviderMeta(
        id="anthropic",
        name="Anthropic",
        litellm_prefix="anthropic",
        supported_tasks=[TaskType.LLM],
        requires_api_key=True,
        description="Anthropic API (Claude)",
    ),
    ProviderMeta(
        id="azure",
        name="Azure OpenAI",
        litellm_prefix="azure",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM, TaskType.RERANKER],
        requires_api_key=True,
        description="Azure OpenAI Service",
    ),
    ProviderMeta(
        id="deepseek",
        name="DeepSeek",
        litellm_prefix="deepseek",
        supported_tasks=[TaskType.LLM],
        requires_api_key=True,
        description="DeepSeek API",
    ),
    ProviderMeta(
        id="groq",
        name="Groq",
        litellm_prefix="groq",
        supported_tasks=[TaskType.LLM],
        requires_api_key=True,
        description="Groq Cloud (fast inference)",
    ),
    ProviderMeta(
        id="together",
        name="Together AI",
        litellm_prefix="together_ai",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="Together AI platform",
    ),
    ProviderMeta(
        id="fireworks",
        name="Fireworks AI",
        litellm_prefix="fireworks_ai",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="Fireworks AI platform",
    ),
    ProviderMeta(
        id="mistral",
        name="Mistral AI",
        litellm_prefix="mistral",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="Mistral AI API",
    ),
    ProviderMeta(
        id="bedrock",
        name="AWS Bedrock",
        litellm_prefix="bedrock",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="AWS Bedrock (Anthropic, Llama, etc.)",
    ),
    ProviderMeta(
        id="vertex_ai",
        name="Google Vertex AI",
        litellm_prefix="vertex_ai",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="Google Vertex AI (Gemini, PaLM)",
    ),
    ProviderMeta(
        id="huggingface",
        name="Hugging Face",
        litellm_prefix="huggingface",
        supported_tasks=[TaskType.EMBEDDING, TaskType.LLM],
        requires_api_key=True,
        description="Hugging Face Inference API",
    ),
]


def get_provider(provider_id: str) -> ProviderMeta | None:
    for p in PROVIDERS:
        if p.id == provider_id:
            return p
    return None


def list_providers(task: TaskType | None = None) -> list[ProviderMeta]:
    if task is None:
        return PROVIDERS
    return [p for p in PROVIDERS if task in p.supported_tasks]
