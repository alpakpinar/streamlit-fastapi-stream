import os
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from graph import build_graph

load_dotenv()

app = FastAPI()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")

class PromptRequest(BaseModel):
    prompt: str


# Build the graph
graph = build_graph()

async def llm_chat_generator(prompt: str) -> AsyncIterator[str]:
    """Streams LLM response using LangGraph."""
    # Stream the response
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        version="v2"
    ):
        # Filter for streaming events from the LLM
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

@app.post("/stream")
async def stream_chat(request: PromptRequest):
    return StreamingResponse(llm_chat_generator(request.prompt), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)