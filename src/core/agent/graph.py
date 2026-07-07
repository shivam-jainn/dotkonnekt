import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.core.chunkers.base import Chunk

logger = logging.getLogger(__name__)

import asyncio
from src.core.chunkers.text import TextChunker
from src.core.embedders.embedder import Embedder
from src.core.parsers.pdf import PDFParser
from src.storage import create_storage
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct
import uuid
from src.configs import settings
class AgentState(TypedDict):
    job_id: str
    files: list[dict]
    
    # Accumulated intermediate state
    chunks: list[Chunk]
    extracted_images: list[dict]
    current_chunk_idx: int
    
    # Chunk-level results
    findings: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    
    # Final report
    overall_score: float
    reconciled_report: dict | None


async def ingest_node(state: AgentState) -> dict:
    job_id = state.get("job_id")
    files = state.get("files", [])
    
    if not files and state.get("chunks"):
        logger.info("[ingest_node] Chunks already provided via initial state. Skipping ingestion for job %s", job_id)
        return {"current_chunk_idx": 0}
        
    logger.info("[ingest_node] Starting ingestion for job %s with %d files", job_id, len(files))
    
    storage = create_storage()
    parser = PDFParser()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    embedder = Embedder(batch_size=100)

    all_chunks = []
    all_images = []
    all_errors = []

    for file_info in files:
        try:
            filename = file_info["filename"]
            storage_path = file_info["storage_path"]
            
            logger.info("[ingest_node] Downloading %s from storage", filename)
            data = await storage.download_bytes(storage_path)
            
            logger.info("[ingest_node] Parsing %s", filename)
            document = await asyncio.to_thread(parser.parse, data, filename)
            
            if hasattr(document, "extracted_images") and document.extracted_images:
                logger.info("[ingest_node] Found %d extracted images in %s", len(document.extracted_images), filename)
                all_images.extend(document.extracted_images)
            
            logger.info("[ingest_node] Chunking %s", filename)
            file_chunks = await asyncio.to_thread(chunker.chunk, document.content, document.metadata)
            
            # Embed chunks for Qdrant storage so we can do Q&A later
            logger.info("[ingest_node] Embedding %d chunks for %s", len(file_chunks), filename)
            embedded = await embedder.embed_chunks(file_chunks)
            
            # Store in Qdrant
            if embedded:
                qdrant = AsyncQdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                )
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=c.embedding,
                        payload={
                            "job_id": job_id,
                            "content": c.content,
                            "index": c.index,
                            "filename": filename,
                        }
                    )
                    for c in embedded
                ]
                try:
                    await qdrant.upsert(
                        collection_name=settings.qdrant_collection,
                        points=points,
                    )
                    logger.info("[ingest_node] Upserted %d chunks to Qdrant collection %s", len(points), settings.qdrant_collection)
                except Exception as e:
                    logger.error("[ingest_node] Qdrant upsert failed: %s", str(e))
                    all_errors.append(f"Qdrant upsert failed for {filename}: {str(e)}")
            
            # Pass the chunks to the state
            all_chunks.extend(embedded)
        except Exception as e:
            err_msg = f"Error ingesting {file_info.get('filename')}: {str(e)}"
            logger.error("[ingest_node] %s", err_msg)
            all_errors.append(err_msg)

    logger.info("[ingest_node] Ingestion completed. Total chunks: %d, Total images: %d, Total errors: %d", len(all_chunks), len(all_images), len(all_errors))
    return {
        "chunks": all_chunks,
        "extracted_images": all_images,
        "current_chunk_idx": 0,
        "errors": all_errors
    }

import base64
import json
import litellm

async def process_chunk_node(state: AgentState) -> dict:
    from src.configs.settings import settings
    chunks = state.get("chunks", [])
    idx = state.get("current_chunk_idx", 0)
    job_id = state.get("job_id")
    
    batch_size = settings.graph_batch_size if settings.graph_batching else 1
    batch_chunks = chunks[idx : idx + batch_size]
    
    if not batch_chunks:
        logger.warning("[process_chunk_node] Job %s index %d out of bounds for chunks size %d", job_id, idx, len(chunks))
        return {}
        
    logger.info("[process_chunk_node] Processing chunks %d-%d/%d for job %s", idx + 1, idx + len(batch_chunks), len(chunks), job_id)
    
    combined_text = "\n\n".join(c["content"] if isinstance(c, dict) else c.content for c in batch_chunks)
    
    prompt = f"""You are a legal document analyst.
Analyze the following document segment and extract:
1. Key obligations
2. Named entities
3. Risky or ambiguous terms

Segment text:
{combined_text}

Respond strictly in JSON format matching this schema:
{{
  "obligations": ["..."],
  "entities": ["..."],
  "risky_terms": ["..."]
}}"""

    messages = [
        {"role": "system", "content": "You are a legal document analyst that outputs valid JSON."},
    ]
    
    content = [{"type": "text", "text": prompt}]
    
    # Attach images to the first chunk/batch if available
    images = state.get("extracted_images", [])
    if images and idx == 0:
        logger.info("[process_chunk_node] Attaching %d images to the first chunk user prompt", min(2, len(images)))
        for img in images[:2]:
            b64 = base64.b64encode(img["bytes"]).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{img['ext']};base64,{b64}"}
            })
            
    messages.append({"role": "user", "content": content})
    
    try:
        from src.core.models.registry import registry
        from src.core.models.providers import TaskType
        import re
        
        llm_kwargs = registry.get_litellm_kwargs(TaskType.LLM)
        if not llm_kwargs:
            llm_kwargs = {"model": "groq/llama-3.3-70b-versatile"}
            
        logger.info("[process_chunk_node] Invoking LLM for chunks %d to %d using model %s", idx + 1, idx + len(batch_chunks), llm_kwargs.get("model"))
        
        max_attempts = 5
        response = None
        for attempt in range(max_attempts):
            try:
                response = await litellm.acompletion(
                    **llm_kwargs,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                break
            except litellm.exceptions.RateLimitError as e:
                if attempt == max_attempts - 1:
                    raise
                err_str = str(e)
                if "Request too large" in err_str or "413" in err_str:
                    logger.warning("[process_chunk_node] Request too large, truncating text and retrying...")
                    content[0]["text"] = content[0]["text"][:int(len(content[0]["text"]) * 0.8)]
                    messages[-1]["content"] = content
                else:
                    match = re.search(r"try again in ([\d.]+)s", err_str)
                    sleep_time = float(match.group(1)) + 1 if match else (20 * (attempt + 1))
                    logger.warning("[process_chunk_node] Rate limit hit, sleeping for %.2fs...", sleep_time)
                    await asyncio.sleep(sleep_time)
            except litellm.exceptions.APIError as e:
                if attempt == max_attempts - 1:
                    raise
                err_str = str(e)
                if "Request too large" in err_str or "413" in err_str:
                    logger.warning("[process_chunk_node] Request too large (APIError), truncating text and retrying...")
                    content[0]["text"] = content[0]["text"][:int(len(content[0]["text"]) * 0.8)]
                    messages[-1]["content"] = content
                else:
                    logger.warning("[process_chunk_node] APIError hit, sleeping for 10s...")
                    await asyncio.sleep(10)
        
        result_text = response.choices[0].message.content
        logger.info("[process_chunk_node] Received LLM response for chunks %d-%d: %s", idx + 1, idx + len(batch_chunks), result_text)
        parsed = json.loads(result_text)
        
        finding = {
            "chunk_idx": idx,
            "obligations": parsed.get("obligations", []),
            "entities": parsed.get("entities", []),
            "risky_terms": parsed.get("risky_terms", [])
        }
        
        return {"findings": [finding]}
    except Exception as e:
        logger.exception("[process_chunk_node] Error processing chunks starting at %d", idx)
        return {"errors": [f"Chunks starting at {idx} error: {str(e)}"]}

async def mcp_tool_node(state: AgentState) -> dict:
    from src.configs.settings import settings
    idx = state.get("current_chunk_idx", 0)
    findings = state.get("findings", [])
    job_id = state.get("job_id")
    chunks = state.get("chunks", [])
    
    logger.info("[mcp_tool_node] Starting MCP analysis step for job %s, chunk index %d, total findings so far: %d", job_id, idx, len(findings))
    
    if findings:
        current_finding = findings[-1]
        risky_terms = current_finding.get("risky_terms", [])
        
        from src.core.agent.mcp_client import call_compliance_tool
        
        scores = []
        for term in risky_terms:
            logger.info("[mcp_tool_node] Calling compliance tool for term: '%s'", term)
            res = await call_compliance_tool(term)
            logger.info("[mcp_tool_node] Compliance result for '%s': score=%s, reasoning=%s", term, res.get("score"), res.get("reasoning"))
            scores.append({
                "term": term,
                "score": res.get("score"),
                "reasoning": res.get("reasoning")
            })
            
        # Update the finding with the MCP scores
        current_finding["mcp_compliance"] = scores
        
    batch_size = settings.graph_batch_size if settings.graph_batching else 1
    actual_batch_size = min(batch_size, len(chunks) - idx)
    return {"current_chunk_idx": idx + actual_batch_size}

def route_next_chunk(state: AgentState) -> str:
    idx = state.get("current_chunk_idx", 0)
    total_chunks = len(state.get("chunks", []))
    if idx < total_chunks:
        logger.info("[route_next_chunk] Next chunk index %d < total chunks %d. Routing to: process_chunk_node", idx, total_chunks)
        return "process_chunk_node"
    logger.info("[route_next_chunk] All %d chunks processed. Routing to: reconcile_node", total_chunks)
    return "reconcile_node"

async def reconcile_node(state: AgentState) -> dict:
    findings = state.get("findings", [])
    job_id = state.get("job_id")
    logger.info("[reconcile_node] Reconciling %d findings for job %s", len(findings), job_id)
    
    all_obligations = []
    all_entities = []
    all_risky_terms = []
    
    for f in findings:
        all_obligations.extend(f.get("obligations", []))
        all_entities.extend(f.get("entities", []))
        all_risky_terms.extend(f.get("risky_terms", []))
        
    # Calculate a mock score for now (based on number of risky terms)
    score = max(0, 100 - (len(all_risky_terms) * 5))
    
    report = {
        "obligations": list(set(all_obligations)),
        "entities": list(set(all_entities)),
        "risky_terms": list(set(all_risky_terms)),
        "score": score
    }
    
    logger.info("[reconcile_node] Reconciled report generated for job %s. Overall score: %d", job_id, score)
    return {
        "overall_score": score,
        "reconciled_report": report
    }

def create_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("ingest_node", ingest_node)
    builder.add_node("process_chunk_node", process_chunk_node)
    builder.add_node("mcp_tool_node", mcp_tool_node)
    builder.add_node("reconcile_node", reconcile_node)
    
    builder.set_entry_point("ingest_node")
    
    builder.add_edge("ingest_node", "process_chunk_node")
    builder.add_edge("process_chunk_node", "mcp_tool_node")
    builder.add_conditional_edges(
        "mcp_tool_node",
        route_next_chunk,
        {
            "process_chunk_node": "process_chunk_node",
            "reconcile_node": "reconcile_node"
        }
    )
    builder.add_edge("reconcile_node", END)
    
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    
    return builder.compile(checkpointer=memory)
