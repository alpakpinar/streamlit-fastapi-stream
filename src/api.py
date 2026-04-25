import os
import json
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from enum import StrEnum

from graph import build_graph
from models import PromptRequest

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")
    
    global graph
    graph = build_graph()
    yield

app = FastAPI(lifespan=lifespan)

NODE_STATUS_MESSAGES = {
    "planning": "Creating a plan...",
    "generate_joke": "Generating a joke...",
}

NODE_OUTPUT_LABELS = {
    "planning": "📋 Plan",
    "generate_joke": "😄 Joke",
}

class GraphEvent(StrEnum):
    ON_CHAT_MODEL_START = "on_chat_model_start"
    ON_CHAT_MODEL_STREAM = "on_chat_model_stream"

def sse_event(event_type: str, data: str) -> str:
    """Format a Server-Sent Event with a typed event field."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

async def llm_chat_generator(prompt: str) -> AsyncIterator[str]:
    """Streams SSE events using LangGraph."""
    announced_nodes: set[str] = set()

    async for event in graph.astream_events(
        {"question": prompt},
        version="v2"
    ):
        node = event.get("metadata", {}).get("langgraph_node")

        # Yield status updates when a new node starts
        if event["event"] == GraphEvent.ON_CHAT_MODEL_START and node in NODE_STATUS_MESSAGES:
            if node not in announced_nodes:
                announced_nodes.add(node)
                yield sse_event("status", NODE_STATUS_MESSAGES[node])

        elif event["event"] == GraphEvent.ON_CHAT_MODEL_STREAM:
            content = event["data"]["chunk"].content
            if not content:
                continue

            # Stream intermediate node tokens so the frontend can show them in dropdowns
            if node in NODE_OUTPUT_LABELS:
                yield sse_event("node_output", f"{node}:{content}")

            # Stream the final answer token by token
            elif node == "answer_question":
                yield sse_event("answer", content)


@app.post("/stream")
async def stream_chat(request: PromptRequest):
    return StreamingResponse(llm_chat_generator(request.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)