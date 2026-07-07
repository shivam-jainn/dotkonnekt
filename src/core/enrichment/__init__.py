"""Metadata enrichment pipeline.

Provides deterministic (regex/heuristic) and optional semantic (LLM)
enrichment of Document IR chunks.
"""

from src.core.enrichment.deterministic import DeterministicEnricher

__all__ = ["DeterministicEnricher"]
