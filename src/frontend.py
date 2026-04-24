import json
import requests
import streamlit as st

API_URL = "http://localhost:8000/stream"

st.title("FastAPI + Streamlit LLM Streamer")

if "messages" not in st.session_state:
    st.session_state.messages = []


def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE line, returning (event_type, data) or (None, None)."""
    if line.startswith("event:"):
        return "event", line[len("event:"):].strip()
    if line.startswith("data:"):
        return "data", line[len("data:"):].removeprefix(" ")
    return None, None


def render_trace(trace_steps: list[dict]) -> None:
    """Render a completed trace as a collapsed expander (used for history replay)."""
    with st.expander(f"Agent trace ({len(trace_steps)} steps)", expanded=False):
        for step in trace_steps:
            duration = f"  `{step['duration_ms']} ms`" if "duration_ms" in step else ""
            st.markdown(f"✅ {step['label']}{duration}")


# ── Chat history ────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("trace"):
            render_trace(message["trace"])
        st.markdown(message["content"])


# ── Live interaction ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me something..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # st.status acts as a live collapsible trace panel — stays visible after completion
        trace_status = st.status("Agent is working...", expanded=True)
        answer_placeholder = st.empty()

        current_event: str | None = None
        answer_text = ""
        trace_steps: list[dict] = []

        with requests.post(API_URL, json={"prompt": prompt}, stream=True) as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                kind, value = parse_sse_line(raw_line)

                if kind == "event":
                    current_event = value

                elif kind == "data":
                    if current_event == "status":
                        # Update the status label while the node is running
                        trace_status.update(label=value, state="running")

                    elif current_event == "trace":
                        # A node finished — append a checkmark with timing inside the trace panel
                        step = json.loads(value)
                        trace_steps.append(step)
                        duration = f"  `{step['duration_ms']} ms`" if "duration_ms" in step else ""
                        trace_status.write(f"✅ {step['label']}{duration}")

                    elif current_event == "answer":
                        # Stream answer tokens into the placeholder below the trace
                        answer_text += value
                        answer_placeholder.markdown(answer_text)

        # Mark trace complete; keep expanded so the user sees the full reasoning
        trace_status.update(label="Done", state="complete", expanded=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "trace": trace_steps,
    })