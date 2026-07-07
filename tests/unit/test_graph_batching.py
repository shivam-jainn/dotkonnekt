import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.core.agent.graph import process_chunk_node, mcp_tool_node, route_next_chunk, AgentState
from src.core.chunkers.base import Chunk
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
    @patch("litellm.acompletion")
    async def test_process_chunk_node_with_batching(self, mock_acompletion):
        # Configure batching: True, size 2
        settings.graph_batching = True
        settings.graph_batch_size = 2

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "obligations": ["Be honest"],
                        "entities": ["Company A"],
                        "risky_terms": ["Indemnity"]
                    })
                )
            )
        ]
        mock_acompletion.return_value = mock_response

        # Build dummy chunks
        chunks = [
            Chunk(content="This is chunk 1", index=0, metadata={}),
            Chunk(content="This is chunk 2", index=1, metadata={}),
            Chunk(content="This is chunk 3", index=2, metadata={}),
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
        
        # Verify LLM was called with concatenated text
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        user_message_content = call_kwargs["messages"][1]["content"][0]["text"]
        assert "This is chunk 1" in user_message_content
        assert "This is chunk 2" in user_message_content
        assert "This is chunk 3" not in user_message_content

        # Verify output findings
        assert "findings" in res1
        assert len(res1["findings"]) == 1
        assert res1["findings"][0]["chunk_idx"] == 0
        assert res1["findings"][0]["obligations"] == ["Be honest"]
        assert res1["findings"][0]["entities"] == ["Company A"]
        assert res1["findings"][0]["risky_terms"] == ["Indemnity"]

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_process_chunk_node_without_batching(self, mock_acompletion):
        # Configure batching: False
        settings.graph_batching = False

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "obligations": ["Pay rent"],
                        "entities": ["Tenant"],
                        "risky_terms": ["Late fee"]
                    })
                )
            )
        ]
        mock_acompletion.return_value = mock_response

        chunks = [
            Chunk(content="This is chunk 1", index=0, metadata={}),
            Chunk(content="This is chunk 2", index=1, metadata={}),
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
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        user_message_content = call_kwargs["messages"][1]["content"][0]["text"]
        assert "This is chunk 1" in user_message_content
        assert "This is chunk 2" not in user_message_content

    @pytest.mark.asyncio
    @patch("src.core.agent.mcp_client.call_compliance_tool")
    async def test_mcp_tool_node_and_increment(self, mock_call_compliance):
        settings.graph_batching = True
        settings.graph_batch_size = 2

        mock_call_compliance.return_value = {"score": 9.5, "reasoning": "Standard term"}

        chunks = [
            Chunk(content="C1", index=0, metadata={}),
            Chunk(content="C2", index=1, metadata={}),
            Chunk(content="C3", index=2, metadata={}),
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
        assert route_next_chunk(state) == "process_chunk_node"

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
