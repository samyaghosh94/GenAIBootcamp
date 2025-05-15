#chat_graph.py

import faiss
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from graph.graph_nodes import generate_response
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from vectorstore.custom_diallab_retriever import DialLabRetriever

from chat_history import load_chat_history


# Update ChatState to match the new structure of using faiss_index, model, and api_key
class ChatState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    faiss_index_path: str
    model: str
    api_key: str


# Function to build the graph
def build_graph(session_id: str, faiss_index_path: str, model: str, api_key: str):
    # Load the chat history (if any) for the given session
    initial_history = load_chat_history(session_id)

    # Load the FAISS index from the file path
    index = faiss.read_index(faiss_index_path)
    # Create an initial state with the required fields
    initial_state: ChatState = {
        "messages": initial_history if initial_history else [], # Use loaded history or start with an empty list
        "session_id": session_id,
        "faiss_index_path": faiss_index_path,
        "model": model,
        "api_key": api_key
    }

    # Initialize the StateGraph with the updated state schema
    graph = StateGraph(state_schema=ChatState)

    # Add the node for generating the response
    graph.add_node("generate_response", generate_response)

    # Set the entry point to the generate_response node
    graph.set_entry_point("generate_response")

    # Create a checkpointer for graph state saving
    checkpointer = MemorySaver()

    # Compile the graph with the checkpointer (don't unpack the result)
    compiled_graph = graph.compile(checkpointer=checkpointer)

    # Return the compiled graph
    return compiled_graph
