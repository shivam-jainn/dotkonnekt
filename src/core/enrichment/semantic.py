"""Optional LLM-based semantic enrichment.

Generates: summary, keywords, obligations, risks, entities, topics, deadlines.
Store alongside deterministic metadata.  Raw text always remains the source
of truth.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.document import Chunk, Document, SemanticMetadata

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """LLM-based enrichment for chunks and documents.

    Set ``enabled=False`` to skip LLM calls entirely (deterministic
    enrichment still runs).
    """

    def __init__(self, enabled: bool = True, model: str | None = None) -> None:
        self.enabled = enabled
        self._model = model

    async def enrich_chunks(self, chunks: list[Chunk]) -> None:
        """Enrich a list of chunks in-place with semantic metadata."""
        if not self.enabled or not chunks:
            return

        for chunk in chunks:
            if not chunk.content.strip():
                continue
            try:
                sm = await self._enrich_single_chunk(chunk.content)
                chunk.semantic_metadata = sm
            except Exception as exc:
                logger.warning(
                    "Semantic enrichment failed for chunk %s: %s", chunk.id, exc
                )

    async def enrich_document(self, document: Document) -> None:
        """Enrich a document's chunks with semantic metadata."""
        if not self.enabled or not document.chunks:
            return
        await self.enrich_chunks(document.chunks)

    async def _enrich_single_chunk(self, text: str) -> SemanticMetadata:
        """Call LLM to extract semantic metadata from chunk text."""
        import litellm

        prompt = f"""Analyze the following document segment and extract metadata.
Respond strictly in JSON:
{{
  "summary": "one-sentence summary",
  "keywords": ["keyword1", "keyword2"],
  "obligations": ["obligation1"],
  "risks": ["risk1"],
  "entities": ["entity1"],
  "topics": ["topic1"],
  "deadlines": ["deadline1"]
}}

Segment:
{text[:3000]}"""

        messages = [
            {
                "role": "system",
                "content": "You are a precise document analyst that outputs valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {}
        if self._model:
            kwargs["model"] = self._model
        else:
            try:
                from src.core.models.providers import TaskType
                from src.core.models.registry import registry

                kwargs = registry.get_litellm_kwargs(TaskType.LLM)
            except Exception:
                kwargs = {"model": "groq/llama-3.3-70b-versatile"}

        provider_id = kwargs.pop("_provider_id", "")
        is_local = provider_id in ("lmstudio", "ollama") or "localhost" in str(kwargs.get("api_base", ""))
        call_kwargs = {**kwargs, "messages": messages}
        if not is_local:
            call_kwargs["response_format"] = {"type": "json_object"}
        response = await litellm.acompletion(**call_kwargs)

        result_text = response.choices[0].message.content or "{}"
        parsed = json.loads(result_text)

        return SemanticMetadata(
            summary=parsed.get("summary"),
            keywords=parsed.get("keywords", []),
            obligations=parsed.get("obligations", []),
            risks=parsed.get("risks", []),
            entities=parsed.get("entities", []),
            topics=parsed.get("topics", []),
            deadlines=parsed.get("deadlines", []),
        )
