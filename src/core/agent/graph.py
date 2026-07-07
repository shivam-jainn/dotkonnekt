import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

import asyncio
import base64
import json
import litellm
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct

from src.configs import settings
from src.core.chunkers.semantic import SemanticChunker, TextChunker
from src.core.clause_graph import build_clause_graph
from src.core.document import Document, Page
from src.core.embedders.embedder import Embedder
from src.core.enrichment.deterministic import DeterministicEnricher
from src.core.parsers.pdf import PDFParser
from src.storage import create_storage


class AgentState(TypedDict):
    job_id: str
    files: list[dict]

    # Accumulated intermediate state
    chunks: list[dict]  # Document IR chunk dicts
    extracted_images: list[dict]
    current_chunk_idx: int

    # Chunk-level results
    findings: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]

    # Final report
    overall_score: float
    reconciled_report: dict | None


def _chunk_to_text(c: dict | object) -> str:
    """Extract content text from a chunk (dict or object)."""
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "")


def _chunk_context_summary(c: dict | object) -> str:
    """Build a context-aware text representation of a chunk with its IR metadata."""
    content = _chunk_to_text(c)
    parts: list[str] = []

    if isinstance(c, dict):
        page = c.get("page")
        section = c.get("section")
        clause = c.get("clause")
        if section:
            parts.append(f"[Section: {section}]")
        if clause:
            parts.append(f"[Clause: {clause}]")
        if page:
            parts.append(f"[Page: {page}]")
    else:
        page = getattr(c, "page", None)
        section = getattr(c, "section", None)
        clause = getattr(c, "clause", None)
        if section:
            parts.append(f"[Section: {section}]")
        if clause:
            parts.append(f"[Clause: {clause}]")
        if page:
            parts.append(f"[Page: {page}]")

    prefix = " ".join(parts)
    return f"{prefix}\n{content}" if prefix else content


async def ingest_node(state: AgentState) -> dict:
    job_id = state.get("job_id")
    files = state.get("files", [])

    if not files and state.get("chunks"):
        logger.info(
            "[ingest_node] Chunks already provided via initial state. Skipping ingestion for job %s",
            job_id,
        )
        return {"current_chunk_idx": 0}

    logger.info(
        "[ingest_node] Starting ingestion for job %s with %d files", job_id, len(files)
    )

    storage = create_storage()
    parser = PDFParser()
    semantic_chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=100, chunk_overlap=100)
    embedder = Embedder(batch_size=100)
    enricher = DeterministicEnricher()

    all_chunks: list[dict] = []
    all_images: list[dict] = []
    all_errors: list[str] = []

    for file_info in files:
        try:
            filename = file_info["filename"]
            storage_path = file_info["storage_path"]

            logger.info("[ingest_node] Downloading %s from storage", filename)
            data = await storage.download_bytes(storage_path)

            logger.info("[ingest_node] Parsing %s to Document IR", filename)
            document = await asyncio.to_thread(parser.parse_to_document, data, filename)
            document.metadata["job_id"] = job_id

            if document.metadata.get("extracted_images"):
                logger.info(
                    "[ingest_node] Found %d extracted images in %s",
                    len(document.metadata["extracted_images"]),
                    filename,
                )
                all_images.extend(document.metadata["extracted_images"])

            logger.info("[ingest_node] Chunking %s with semantic chunker", filename)
            doc_chunks = await asyncio.to_thread(semantic_chunker.chunk_document, document)

            logger.info("[ingest_node] Enriching metadata for %s", filename)
            await asyncio.to_thread(enricher.enrich_document, document)

            logger.info("[ingest_node] Building clause graph for %s", filename)
            await asyncio.to_thread(build_clause_graph, document)

            # Embed chunks for Qdrant storage
            logger.info(
                "[ingest_node] Embedding %d chunks for %s", len(doc_chunks), filename
            )
            _, embeddings = await embedder.embed_document_chunks(doc_chunks)

            # Store in Qdrant with Document IR payload
            if doc_chunks and embeddings:
                qdrant = AsyncQdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                )
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "job_id": job_id,
                            "filename": filename,
                            **chunk.to_payload(),
                        },
                    )
                    for chunk, emb in zip(doc_chunks, embeddings)
                ]
                try:
                    await qdrant.upsert(
                        collection_name=settings.qdrant_collection,
                        points=points,
                    )
                    logger.info(
                        "[ingest_node] Upserted %d chunks to Qdrant collection %s",
                        len(points),
                        settings.qdrant_collection,
                    )
                except Exception as e:
                    logger.error("[ingest_node] Qdrant upsert failed: %s", str(e))
                    all_errors.append(
                        f"Qdrant upsert failed for {filename}: {str(e)}"
                    )

            # Pass chunks as dicts to state
            all_chunks.extend(
                {**chunk.to_payload(), "id": chunk.id} for chunk in doc_chunks
            )
        except Exception as e:
            err_msg = f"Error ingesting {file_info.get('filename')}: {str(e)}"
            logger.error("[ingest_node] %s", err_msg)
            all_errors.append(err_msg)

    logger.info(
        "[ingest_node] Ingestion completed. Total chunks: %d, Total images: %d, Total errors: %d",
        len(all_chunks),
        len(all_images),
        len(all_errors),
    )
    return {
        "chunks": all_chunks,
        "extracted_images": all_images,
        "current_chunk_idx": 0,
        "errors": all_errors,
    }


async def risk_analysis_node(state: AgentState) -> dict:
    chunks = state.get("chunks", [])
    idx = state.get("current_chunk_idx", 0)
    job_id = state.get("job_id")

    from src.configs.settings import settings
    batch_size = settings.graph_batch_size if settings.graph_batching else 1
    batch_chunks = chunks[idx : idx + batch_size]

    if not batch_chunks:
        logger.warning(
            "[risk_analysis_node] Job %s index %d out of bounds for chunks size %d",
            job_id,
            idx,
            len(chunks),
        )
        return {}

    logger.info(
        "[risk_analysis_node] Processing chunks %d-%d/%d for job %s",
        idx + 1,
        idx + len(batch_chunks),
        len(chunks),
        job_id,
    )

    findings = []
    for i, c in enumerate(batch_chunks):
        chunk_idx = idx + i
        dm_dict = c.get("derived_metadata") or {}
        
        findings.append({
            "chunk_idx": chunk_idx,
            "obligations": dm_dict.get("obligations", []),
            "entities": dm_dict.get("entities", []),
            "risky_terms": dm_dict.get("risks", []),
            "party_sentences": dm_dict.get("party_sentences", {}),
            "obligations_by_party": dm_dict.get("obligations_by_party", []),
            "risks_by_party": dm_dict.get("risks_by_party", []),
        })

    return {"findings": findings}

process_chunk_node = risk_analysis_node



async def mcp_tool_node(state: AgentState) -> dict:
    from src.configs.settings import settings

    idx = state.get("current_chunk_idx", 0)
    findings = state.get("findings", [])
    job_id = state.get("job_id")
    chunks = state.get("chunks", [])

    logger.info(
        "[mcp_tool_node] Starting MCP analysis step for job %s, chunk index %d, total findings so far: %d",
        job_id,
        idx,
        len(findings),
    )

    if findings:
        current_finding = findings[-1]
        risky_terms = current_finding.get("risky_terms", [])

        from src.core.agent.mcp_client import call_compliance_tool

        scores = []
        for term in risky_terms:
            logger.info("[mcp_tool_node] Calling compliance tool for term: '%s'", term)
            res = await call_compliance_tool(term)
            logger.info(
                "[mcp_tool_node] Compliance result for '%s': score=%s, reasoning=%s",
                term,
                res.get("score"),
                res.get("reasoning"),
            )
            scores.append(
                {
                    "term": term,
                    "score": res.get("score"),
                    "reasoning": res.get("reasoning"),
                }
            )

        # Update the finding with the MCP scores
        current_finding["mcp_compliance"] = scores

    batch_size = settings.graph_batch_size if settings.graph_batching else 1
    actual_batch_size = min(batch_size, len(chunks) - idx)
    return {"current_chunk_idx": idx + actual_batch_size}


def route_next_chunk(state: AgentState) -> str:
    idx = state.get("current_chunk_idx", 0)
    total_chunks = len(state.get("chunks", []))
    if idx < total_chunks:
        logger.info(
            "[route_next_chunk] Next chunk index %d < total chunks %d. Routing to: risk_analysis_node",
            idx,
            total_chunks,
        )
        return "risk_analysis_node"
    logger.info(
        "[route_next_chunk] All %d chunks processed. Routing to: reconcile_node",
        total_chunks,
    )
    return "reconcile_node"


async def reconcile_node(state: AgentState) -> dict:
    findings = state.get("findings", [])
    job_id = state.get("job_id")
    logger.info("[reconcile_node] Reconciling %d findings for job %s", len(findings), job_id)

    all_obligations = []
    all_entities = []
    all_risky_terms = []
    parties = {}

    for f in findings:
        all_obligations.extend(f.get("obligations", []))
        all_entities.extend(f.get("entities", []))
        all_risky_terms.extend(f.get("risky_terms", []))

        ob_by_party = f.get("obligations_by_party", [])
        for ob in ob_by_party:
            role = ob.get("party_role")
            if role and role != "General":
                parties.setdefault(role, {"obligations": [], "risks": []})
                if ob.get("text") not in parties[role]["obligations"]:
                    parties[role]["obligations"].append(ob.get("text"))

        rk_by_party = f.get("risks_by_party", [])
        for rk in rk_by_party:
            role = rk.get("party_role")
            if role and role != "General":
                parties.setdefault(role, {"obligations": [], "risks": []})
                if rk.get("description") not in parties[role]["risks"]:
                    parties[role]["risks"].append(rk.get("description"))

    # Calculate a mock score for now (based on number of risky terms)
    score = max(0, 100 - (len(all_risky_terms) * 5))

    report = {
        "parties": parties,
        "obligations": list(set(all_obligations)),
        "entities": list(set(all_entities)),
        "risky_terms": list(set(all_risky_terms)),
        "score": score,
    }

    logger.info(
        "[reconcile_node] Reconciled report generated for job %s. Overall score: %d",
        job_id,
        score,
    )
    return {"overall_score": score, "reconciled_report": report}


def create_graph():
    builder = StateGraph(AgentState)

    builder.add_node("ingest_node", ingest_node)
    builder.add_node("risk_analysis_node", risk_analysis_node)
    builder.add_node("mcp_tool_node", mcp_tool_node)
    builder.add_node("reconcile_node", reconcile_node)

    builder.set_entry_point("ingest_node")

    builder.add_edge("ingest_node", "risk_analysis_node")
    builder.add_edge("risk_analysis_node", "mcp_tool_node")
    builder.add_conditional_edges(
        "mcp_tool_node",
        route_next_chunk,
        {
            "risk_analysis_node": "risk_analysis_node",
            "reconcile_node": "reconcile_node",
        },
    )
    builder.add_edge("reconcile_node", END)

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()

    return builder.compile(checkpointer=memory)
