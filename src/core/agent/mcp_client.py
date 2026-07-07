import sys
import logging
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import json

logger = logging.getLogger(__name__)

async def call_compliance_tool(clause_text: str, category: str = "general") -> dict:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool("calculate_compliance_score", {
                    "clause_text": clause_text,
                    "category": category
                })
                
                if result.content and len(result.content) > 0:
                    return json.loads(result.content[0].text)
                return {"score": 50, "reasoning": "No content returned"}
    except Exception as e:
        logger.error(f"MCP Tool call failed: {e}")
        return {"score": 50, "reasoning": f"MCP Error: {str(e)}"}
