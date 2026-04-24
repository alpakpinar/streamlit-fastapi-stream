import os
import json
import time
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
    
    has_azure = os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
    has_openai = os.getenv("OPENAI_API_KEY")

    if not has_azure and not has_openai:
        raise EnvironmentError(
            "No LLM credentials found. Set either:\n"
            "  - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (Azure OpenAI)\n"
            "  - OPENAI_API_KEY  (native OpenAI)"
        )
    
    global graph
    graph = build_graph()
    yield

app = FastAPI(lifespan=lifespan)

# Status messages shown while a node is running (on_chat_model_start)
NODE_STATUS_MESSAGES = {
    "planning": "Creating a plan...",
    "generate_joke": "Generating a joke...",
}

# Trace labels emitted when a node completes (on_chain_end)
NODE_TRACE_LABELS = {
    "planning": "Plan created",
    "generate_joke": "Joke generated",
    "answer_question": "Answer generated",
}

class GraphEvent(StrEnum):
    ON_CHAT_MODEL_START = "on_chat_model_start"
    ON_CHAT_MODEL_STREAM = "on_chat_model_stream"
    ON_CHAIN_START = "on_chain_start"
    ON_CHAIN_END = "on_chain_end"

def sse_event(event_type: str, data: str) -> str:
    """Format a Server-Sent Event with a typed event field."""
    return f"event: {event_type}\ndata: {data}\n\n"

async def llm_chat_generator(prompt: str) -> AsyncIterator[str]:
    """Streams SSE events using LangGraph.

    Event types emitted:
      status  — short human-readable label while a node is running
      trace   — JSON payload emitted when a node completes (includes duration_ms)
      answer  — token-by-token chunks from the final answer node
    """
    announced_nodes: set[str] = set()
    node_start_times: dict[str, float] = {}

    async for event in graph.astream_events(
        {"question": prompt},
        version="v2"
    ):
        node = event.get("metadata", {}).get("langgraph_node")
        event_name = event["event"]
        run_id = event.get("run_id")

        # Record start time when a tracked node begins
        if event_name == GraphEvent.ON_CHAIN_START and event.get("name") in NODE_TRACE_LABELS:
            node_start_times[run_id] = time.monotonic()

        # Emit a status banner when an LLM call starts inside a known node
        elif event_name == GraphEvent.ON_CHAT_MODEL_START and node in NODE_STATUS_MESSAGES:
            if node not in announced_nodes:
                announced_nodes.add(node)
                yield sse_event("status", NODE_STATUS_MESSAGES[node])

        # Emit a trace record when a known node finishes
        elif event_name == GraphEvent.ON_CHAIN_END and event.get("name") in NODE_TRACE_LABELS:
            duration_ms = round((time.monotonic() - node_start_times.pop(run_id, time.monotonic())) * 1000)
            trace_payload = json.dumps({
                "node": event["name"],
                "label": NODE_TRACE_LABELS[event["name"]],
                "duration_ms": duration_ms,
            })
            yield sse_event("trace", trace_payload)

        # Stream answer tokens from the final node only
        elif event_name == GraphEvent.ON_CHAT_MODEL_STREAM and node == "answer_question":
            content = event["data"]["chunk"].content
            if content:
                yield sse_event("answer", content)


@app.post("/stream")
async def stream_chat(request: PromptRequest):
    return StreamingResponse(llm_chat_generator(request.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)