import streamlit as st
import requests

st.title("FastAPI + Streamlit LLM Streamer")

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_streaming_response(prompt):
    """
    Generator that fetches data from FastAPI and yields chunks.
    """
    url = "http://localhost:8000/stream"
    # stream=True is critical for keeping the connection open
    with requests.post(url, json={"prompt": prompt}, stream=True) as response:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Simple chat input
if prompt := st.chat_input("Ask me something..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        # st.write_stream consumes our generator and updates the UI in real-time
        response_text = st.write_stream(get_streaming_response(prompt))
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})