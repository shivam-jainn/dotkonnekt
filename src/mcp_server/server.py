import sys
import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

logger = logging.getLogger("mcp_server")

app = Server("compliance_server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculate_compliance_score",
            description="Evaluates a legal clause against compliance baselines and returns a score and reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "clause_text": {"type": "string"},
                    "category": {"type": "string"}
                },
                "required": ["clause_text", "category"]
            }
        )
    ]

@app.call_tool()
async def call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "calculate_compliance_score":
        clause_text = arguments.get("clause_text", "") if arguments else ""
        category = arguments.get("category", "general") if arguments else "general"
        
        score = 100
        reasoning = "Clause looks standard."
        
        clause_lower = clause_text.lower()
        if "unlimited liability" in clause_lower or "will not indemnify" in clause_lower or "indemnify" in clause_lower:
            score = 20
            reasoning = "Risky indemnity or liability clause detected."
        elif "best effort" in clause_lower or "sole discretion" in clause_lower:
            score = 50
            reasoning = "Ambiguous term found. Consider revising."
            
        result = json.dumps({
            "score": score,
            "reasoning": reasoning,
            "category": category
        })
        
        return [
            types.TextContent(
                type="text",
                text=result
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    import asyncio
    asyncio.run(run())
