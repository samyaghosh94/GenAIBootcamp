# main.py
import asyncio
import os
import json
import subprocess
import sys
import uuid
from dotenv import load_dotenv
import streamlit as st

from chat_history import save_chat_history
from config import SCREENSHOT_DIR, DOCSTORE_PATH, FAISS_INDEX_PATH
import context_data_setup
from parsers.omniparser_client import parse_image_with_retries
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings
from vectorstore.custom_diallab_retriever import DialLabRetriever
from vectorstore.loader import load_or_build_vectorstore
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from graph.chat_graph import build_graph  # Your build_graph implementation
# Set the global retriever used inside generate_response
import global_retriever

# Load environment variables
load_dotenv()

# Function to load documents
def load_or_parse_documents():
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            texts = [doc["page_content"] for doc in json.load(f)]
    return texts

# Streamlit UI setup
st.set_page_config(page_title="Ustora WebApp Chatbot")
st.title("🤖 Ustora ChatBot: Ask about your App Screens!")

# Generate a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Parse or load documents
parsed_texts = load_or_parse_documents()
all_chunks = [Document(page_content=text) for text in parsed_texts]

# 2. Prepare Embeddings & Vectorstore
embeddings = DialLabEmbeddings(
    model=os.getenv("DIAL_LAB_MODEL"),
    api_key=os.getenv("DIAL_LAB_KEY"),
    base_url=os.getenv("DIAL_LAB_BASE_URL")
)

# 3. Load or Build FAISS Index and Vectorstore
vectorstore = load_or_build_vectorstore(all_chunks, embeddings)

# 4. Build retriever (which internally uses the FAISS index)
retriever = DialLabRetriever(
    model=embeddings.model,
    api_key=embeddings.api_key,
    base_url=os.getenv("DIAL_LAB_BASE_URL"),
    faiss_index=vectorstore
)

# Set global retriever for generate_response
global_retriever.retriever = retriever
print("Global retriever set:", global_retriever.retriever)

# Save the FAISS index path
faiss_index_path = FAISS_INDEX_PATH


# 5. Build graph (compile only the graph)
graph = build_graph(
    st.session_state.session_id,
    faiss_index_path=faiss_index_path,
    model=embeddings.model,
    api_key=embeddings.api_key
)

# 6. Chat input field
user_input = st.chat_input("Ask something about the app...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    state = {
        "messages": st.session_state.chat_history,
        "session_id": st.session_state.session_id,
        "faiss_index_path": faiss_index_path,
        "model": embeddings.model,
        "api_key": embeddings.api_key
    }

    # Print the state to understand what we are passing to graph.invoke()
    print("State being passed into graph.invoke():", state)

    # Ensure the retriever is passed correctly to the state
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": st.session_state.session_id}}
    )

    st.session_state.chat_history = result["messages"]

    # Save the updated chat history to a file
    save_chat_history(st.session_state.session_id, st.session_state.chat_history)

# 7. Show chat messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)
