import streamlit as st
import requests
import json
from chat_message import ChatMessage, NODE_OUTPUT_LABELS

API_URL = "http://localhost:8000/stream"

st.title("FastAPI + Streamlit LLM Streamer")
if "messages" not in st.session_state:
    st.session_state.messages: list[ChatMessage] = []

def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE line, returning (event_type, data) or (None, None)."""
    if line.startswith("event:"):
        return "event", line[len("event:"):].strip()
    if line.startswith("data:"):
        return "data", json.loads(line[len("data:"):].removeprefix(" "))
    return None, None

def stream_response(prompt, status_placeholder, node_outputs: dict):
    """Consume the SSE stream, show status in a box, collect node outputs, yield answer chunks."""
    current_event = None

    with requests.post(API_URL, json={"prompt": prompt}, stream=True) as response:
        for raw_line in response.iter_lines(decode_unicode=True):
            kind, value = parse_sse_line(raw_line)

            if kind == "event":
                current_event = value
            elif kind == "data":
                if current_event == "status":
                    status_placeholder.status(value, state="running")
                elif current_event == "node_output":
                    # value is formatted as "node_name:token"
                    node, _, token = value.partition(":")
                    node_outputs[node] = node_outputs.get(node, "") + token
                elif current_event == "answer":
                    status_placeholder.empty()
                    yield value


for message in st.session_state.messages:
    message.render()

if prompt := st.chat_input("Ask me something..."):
    user_message = ChatMessage(role="user", content=prompt)
    user_message.render()

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        node_outputs: dict = {}
        answer_placeholder = st.empty()
        response_text = ""
        for chunk in stream_response(prompt, status_placeholder, node_outputs):
            response_text += chunk
            answer_placeholder.markdown(response_text + "▌")
        answer_placeholder.markdown(response_text)
        for node, content in node_outputs.items():
            label = NODE_OUTPUT_LABELS.get(node, node)
            with st.expander(label):
                st.markdown(content)

    st.session_state.messages.append(user_message)
    st.session_state.messages.append(ChatMessage(role="assistant", content=response_text, node_outputs=node_outputs))