from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# Define the chatbot function for LangGraph
def chatbot(state: MessagesState):

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper option
        streaming=True,
        temperature=0.0
    )
    return {"messages": [llm.invoke(state["messages"])]}

def build_graph():
    """Builds the LangGraph state graph for the chatbot."""
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    
    return graph