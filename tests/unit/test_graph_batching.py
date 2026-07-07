import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.core.agent.graph import process_chunk_node, mcp_tool_node, route_next_chunk, AgentState
from src.configs.settings import settings


@pytest.mark.unit
class TestGraphBatching:
    @pytest.fixture(autouse=True)
    def setup_settings(self):
        # Save original settings
        self.orig_batching = settings.graph_batching
        self.orig_batch_size = settings.graph_batch_size
        yield
        # Restore settings
        settings.graph_batching = self.orig_batching
        settings.graph_batch_size = self.orig_batch_size

    @pytest.mark.asyncio
    async def test_process_chunk_node_with_batching(self):
        # Configure batching: True, size 2
        settings.graph_batching = True
        settings.graph_batch_size = 2

        # Build dummy chunks as dicts (Document IR representation) with derived_metadata
        chunks = [
            {
                "content": "This is chunk 1", 
                "index": 0, 
                "metadata": {}, 
                "page": 1, 
                "section": None, 
                "clause": None,
                "derived_metadata": {
                    "obligations": ["Be honest"],
                    "entities": ["Company A"],
                    "risks": ["Indemnity"]
                }
            },
            {
                "content": "This is chunk 2", 
                "index": 1, 
                "metadata": {}, 
                "page": 1, 
                "section": None, 
                "clause": None,
                "derived_metadata": {
                    "obligations": [],
                    "entities": [],
                    "risks": []
                }
            },
            {
                "content": "This is chunk 3", 
                "index": 2, 
                "metadata": {}, 
                "page": 1, 
                "section": None, 
                "clause": None,
                "derived_metadata": {
                    "obligations": [],
                    "entities": [],
                    "risks": []
                }
            },
        ]

        state: AgentState = {
            "job_id": "job-abc",
            "files": [],
            "chunks": chunks,
            "extracted_images": [],
            "current_chunk_idx": 0,
            "findings": [],
            "errors": [],
            "overall_score": 0.0,
            "reconciled_report": None
        }

        # Process first batch (chunks 1 and 2)
        res1 = await process_chunk_node(state)

        # Verify output findings
        assert "findings" in res1
        assert len(res1["findings"]) == 2
        assert res1["findings"][0]["chunk_idx"] == 0
        assert res1["findings"][0]["obligations"] == ["Be honest"]
        assert res1["findings"][0]["entities"] == ["Company A"]
        assert res1["findings"][0]["risky_terms"] == ["Indemnity"]

    @pytest.mark.asyncio
    async def test_process_chunk_node_without_batching(self):
        # Configure batching: False
        settings.graph_batching = False

        chunks = [
            {
                "content": "This is chunk 1", 
                "index": 0, 
                "metadata": {}, 
                "page": 1, 
                "section": None, 
                "clause": None,
                "derived_metadata": {
                    "obligations": ["Pay rent"],
                    "entities": ["Tenant"],
                    "risks": ["Late fee"]
                }
            },
            {
                "content": "This is chunk 2", 
                "index": 1, 
                "metadata": {}, 
                "page": 1, 
                "section": None, 
                "clause": None,
                "derived_metadata": {
                    "obligations": [],
                    "entities": [],
                    "risks": []
                }
            },
        ]

        state: AgentState = {
            "job_id": "job-xyz",
            "files": [],
            "chunks": chunks,
            "extracted_images": [],
            "current_chunk_idx": 0,
            "findings": [],
            "errors": [],
            "overall_score": 0.0,
            "reconciled_report": None
        }

        res = await process_chunk_node(state)
        assert "findings" in res
        assert len(res["findings"]) == 1
        assert res["findings"][0]["chunk_idx"] == 0
        assert res["findings"][0]["obligations"] == ["Pay rent"]
        assert res["findings"][0]["entities"] == ["Tenant"]
        assert res["findings"][0]["risky_terms"] == ["Late fee"]

    @pytest.mark.asyncio
    @patch("src.core.agent.mcp_client.call_compliance_tool")
    async def test_mcp_tool_node_and_increment(self, mock_call_compliance):
        settings.graph_batching = True
        settings.graph_batch_size = 2

        mock_call_compliance.return_value = {"score": 9.5, "reasoning": "Standard term"}

        chunks = [
            {"content": "C1", "index": 0, "metadata": {}, "page": 1, "section": None, "clause": None},
            {"content": "C2", "index": 1, "metadata": {}, "page": 1, "section": None, "clause": None},
            {"content": "C3", "index": 2, "metadata": {}, "page": 1, "section": None, "clause": None},
        ]

        # Let's say process_chunk_node just returned findings for the first batch
        state: AgentState = {
            "job_id": "job-123",
            "files": [],
            "chunks": chunks,
            "extracted_images": [],
            "current_chunk_idx": 0,
            "findings": [{
                "chunk_idx": 0,
                "obligations": [],
                "entities": [],
                "risky_terms": ["Indemnity"]
            }],
            "errors": [],
            "overall_score": 0.0,
            "reconciled_report": None
        }

        res = await mcp_tool_node(state)

        # Verify compliance tool was called for the latest finding's terms
        mock_call_compliance.assert_called_once_with("Indemnity")
        assert state["findings"][0]["mcp_compliance"][0]["score"] == 9.5

        # Verify index increments by batch size (min(2, 3 - 0) = 2)
        assert res == {"current_chunk_idx": 2}

        # Route next chunk check
        state["current_chunk_idx"] = res["current_chunk_idx"]
        assert route_next_chunk(state) == "risk_analysis_node"

        # Now let's do the second batch starting at index 2 (last remaining chunk)
        state["findings"].append({
            "chunk_idx": 2,
            "obligations": [],
            "entities": [],
            "risky_terms": []
        })
        mock_call_compliance.reset_mock()
        res2 = await mcp_tool_node(state)

        # Batch size for idx 2: min(2, 3 - 2) = 1
        assert res2 == {"current_chunk_idx": 3}
        state["current_chunk_idx"] = res2["current_chunk_idx"]
        assert route_next_chunk(state) == "reconcile_node"
