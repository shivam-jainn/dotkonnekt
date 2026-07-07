import logging
import json
from typing import Any, Dict, List
from langchain_core.callbacks.base import BaseCallbackHandler
from src.database import db
from sqlalchemy import text

logger = logging.getLogger(__name__)

class StructuredTraceCollector(BaseCallbackHandler):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.traces = []

    def _record(self, event_type: str, data: dict):
        trace_event = {
            "event_type": event_type,
            "data": data
        }
        self.traces.append(trace_event)
        
        # Log to terminal for better visibility
        preview = str(data)
        if len(preview) > 300:
            preview = preview[:300] + "..."
        logger.info("[trace_collector] %s: %s", event_type, preview)
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        # Avoid circular serialization issues by simplifying inputs
        safe_inputs = str(inputs)[:500] if inputs else ""
        self._record("chain_start", {"name": (serialized or {}).get("name", "unknown"), "inputs": safe_inputs})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        safe_outputs = str(outputs)[:500] if outputs else ""
        self._record("chain_end", {"outputs": safe_outputs})

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self._record("llm_start", {"prompts": prompts, "model": kwargs.get("invocation_params", {}).get("model")})

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        self._record("llm_end", {"response": response.dict()})
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        self._record("tool_start", {"name": (serialized or {}).get("name", "unknown"), "input": input_str})

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        self._record("tool_end", {"output": str(output)})
        
    async def flush_to_db(self):
        try:
            async with db.pool() as session:
                # Store traces in the jobs table metadata JSONB
                await session.execute(
                    text("UPDATE jobs SET metadata = jsonb_set(metadata, '{traces}', cast(:traces as jsonb)) WHERE id = :job_id"),
                    {"traces": json.dumps(self.traces), "job_id": self.job_id}
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to flush traces: {e}")
