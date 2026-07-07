import json
import logging
from typing import AsyncIterator, TypedDict

import litellm
from langgraph.graph import END, StateGraph
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from src.configs import settings
from src.core.embedders.embedder import Embedder
from src.core.models.providers import TaskType
from src.core.models.registry import registry

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



class QAState(TypedDict):
    job_id: str
    query: str
    top_k: int
    collection: str
    context_chunks: list[dict]
    answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def retrieve_context_node(state: QAState) -> dict:
    job_id = state["job_id"]
    query = state["query"]
    top_k = state.get("top_k", 5)
    collection = state.get("collection", settings.qdrant_collection)

    qdrant = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=False,
        check_compatibility=False,
    )

    embedder = Embedder()
    try:
        query_vector = await embedder.embed_query(query)
    except Exception as exc:
        logger.error("[retrieve_context_node] Failed to embed query: %s", exc)
        return {"context_chunks": []}

    try:
        search_result = await qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        chunks = [
            {
                "content": hit.payload.get("content", ""),
                "index": hit.payload.get("index", 0),
                "score": hit.score,
            }
            for hit in search_result.points
        ]

        logger.info(
            "[retrieve_context_node] Retrieved %d chunks for job %s", len(chunks), job_id
        )
        return {"context_chunks": chunks}

    except Exception as exc:
        logger.error("[retrieve_context_node] Qdrant search failed: %s", exc)
        return {"context_chunks": []}


async def generate_answer_node(state: QAState) -> dict:
    query = state["query"]
    chunks = state.get("context_chunks", [])

    if not chunks:
        return {"answer": "No relevant context found in the document for your query."}

    context_text = "\n\n".join(
        f"[Chunk {c['index']}] {c['content']}" for c in chunks
    )

    prompt = (
        "Answer the query using ONLY the context below. Be concise — direct sentences, no filler. "
        "Cite sources as [Chunk X]. If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Query: {query}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document analyst. Output only the answer — no thinking, "
                "no reasoning steps, no preamble. Be direct and concise."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    llm_kwargs = registry.get_litellm_kwargs(TaskType.LLM)
    if not llm_kwargs:
        llm_kwargs = {"model": "groq/llama-3.3-70b-versatile"}

    try:
        response = await litellm.acompletion(**llm_kwargs, messages=messages)
        answer = response.choices[0].message.content or ""
        answer = _strip_think_tags(answer)
        return {"answer": answer}
    except Exception as exc:
        logger.error("[generate_answer_node] LLM call failed: %s", exc)
        return {"answer": f"Failed to generate answer: {str(exc)}"}


def create_qa_graph():
    """Return a compiled QA StateGraph. Kept for backward-compat."""
    builder = StateGraph(QAState)
    builder.add_node("retrieve", retrieve_context_node)
    builder.add_node("generate", generate_answer_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


async def run_qa(
    job_id: str,
    query: str,
    top_k: int = 5,
    collection: str | None = None,
) -> dict:
    """
    Run the full QA pipeline synchronously and return:
        {"answer": str, "context_chunks": list[dict]}
    """
    app = create_qa_graph()
    print("intializing QA graph for job and app : ", job_id)
    print(f"[run_qa] Running QA for job {job_id} with query: {query}")
    initial_state: QAState = {
        "job_id": job_id,
        "query": query,
        "top_k": top_k,
        "collection": collection or settings.qdrant_collection,
        "context_chunks": [],
        "answer": "",
    }
    print(f"[run_qa] Initial state: {initial_state}")
    result = await app.ainvoke(initial_state)
    print(f"[run_qa] Final result for job {job_id}: {result}")

    return {
        "answer": result.get("answer", ""),
        "context_chunks": result.get("context_chunks", []),
    }


async def run_qa_stream(
    job_id: str,
    query: str,
    top_k: int = 5,
    collection: str | None = None,
) -> AsyncIterator[dict]:
    """
    Stream the QA pipeline as SSE events:
      - event='context'  data=JSON list of retrieved chunks (emitted first)
      - event='token'    data=<text token>   (one per streamed LLM token)
      - event='done'     data=JSON {"answer": <full answer>}
    """
    resolved_collection = collection or settings.qdrant_collection

    # --- Step 1: retrieve context ---
    context_state = await retrieve_context_node(
        {
            "job_id": job_id,
            "query": query,
            "top_k": top_k,
            "collection": resolved_collection,
            "context_chunks": [],
            "answer": "",
        }
    )
    chunks = context_state.get("context_chunks", [])

    yield {
        "event": "context",
        "data": json.dumps(
            [
                {"index": c.get("index", 0), "content": c.get("content", ""), "score": c.get("score")}
                for c in chunks
            ]
        ),
    }

    # --- Step 2: stream answer from LLM ---
    if not chunks:
        yield {
            "event": "done",
            "data": json.dumps({"answer": "No relevant context found in the document for your query."}),
        }
        return

    context_text = "\n\n".join(
        f"[Chunk {c['index']}] {c['content']}" for c in chunks
    )
    prompt = (
        "Answer the query using ONLY the context below. Be concise — direct sentences, no filler. "
        "Cite sources as [Chunk X]. If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Query: {query}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document analyst. Output only the answer — no thinking, "
                "no reasoning steps, no preamble. Be direct and concise."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    llm_kwargs = registry.get_litellm_kwargs(TaskType.LLM)
    if not llm_kwargs:
        llm_kwargs = {"model": "groq/llama-3.3-70b-versatile"}

    full_answer = ""
    try:
        response = await litellm.acompletion(**llm_kwargs, messages=messages, stream=True)
        async for chunk in response:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if token:
                full_answer += token
                yield {"event": "token", "data": token}
    except Exception as exc:
        logger.error("[run_qa_stream] Streaming LLM call failed: %s", exc)
        full_answer = f"Failed to generate answer: {str(exc)}"

    yield {
        "event": "done",
        "data": json.dumps({"answer": _strip_think_tags(full_answer)}),
    }
