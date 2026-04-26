import streamlit as st

def sidebar():
    with st.sidebar:
        st.title("LLM Streamer")
        st.markdown(
            "This app demonstrates streaming responses from a FastAPI backend."
        )
        with st.expander("About", expanded=False):
            st.markdown(
                """
                **Tech Stack**:
                - FastAPI for the backend API
                - Streamlit for the frontend UI
                """
            )
        