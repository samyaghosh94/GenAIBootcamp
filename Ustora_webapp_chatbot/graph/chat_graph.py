# chat_graph.py

import faiss
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from graph.graph_nodes import generate_response
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langsmith import traceable
from vectorstore.custom_diallab_retriever import DialLabRetriever

from chat_history import load_chat_history

class ChatState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    faiss_index_path: str
    model: str
    api_key: str

@traceable()
def build_graph(session_id: str, faiss_index_path: str, model: str, api_key: str):
    initial_history = load_chat_history(session_id)
    # index = faiss.read_index(faiss_index_path)
    #
    # initial_state: ChatState = {
    #     "messages": initial_history if initial_history else [],
    #     "session_id": session_id,
    #     "faiss_index_path": faiss_index_path,
    #     "model": model,
    #     "api_key": api_key
    # }

    graph = StateGraph(state_schema=ChatState)
    graph.add_node("generate_response", generate_response)
    graph.set_entry_point("generate_response")
    checkpointer = MemorySaver()

    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph
