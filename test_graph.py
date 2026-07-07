import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks.base import BaseCallbackHandler

class MyTracer(BaseCallbackHandler):
    pass

class State(TypedDict):
    val: int

def node(state: State):
    return {"val": state["val"] + 1}

async def main():
    b = StateGraph(State)
    b.add_node("n", node)
    b.set_entry_point("n")
    b.add_edge("n", END)
    app = b.compile(checkpointer=MemorySaver())
    
    tracer = MyTracer()
    config = {"callbacks": [tracer], "configurable": {"thread_id": "123"}}
    print(await app.ainvoke({"val": 0}, config=config))

asyncio.run(main())
