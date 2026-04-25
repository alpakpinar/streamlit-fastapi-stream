import streamlit as st
from dataclasses import dataclass, field

NODE_OUTPUT_LABELS = {
    "planning": "📋 Plan",
    "generate_joke": "😄 Joke",
}


@dataclass
class ChatMessage:
    """Represents a single chat message, optionally with node outputs."""
    role: str
    content: str
    node_outputs: dict[str, str] = field(default_factory=dict)

    def render(self):
        """Renders the chat message in Streamlit, including any node outputs in expanders."""
        with st.chat_message(self.role):
            if self.node_outputs:
                for node, content in self.node_outputs.items():
                    label = NODE_OUTPUT_LABELS.get(node, node)
                    with st.expander(label):
                        st.markdown(content)
            
            st.markdown(self.content)
