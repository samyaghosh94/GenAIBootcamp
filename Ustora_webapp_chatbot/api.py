import json
import os
import uuid
from typing import List

from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from config import FAISS_INDEX_PATH, SCREENSHOT_DIR, DOCSTORE_PATH
from vectorstore.loader import load_or_build_vectorstore
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings
from vectorstore.custom_diallab_retriever import DialLabRetriever
from langchain.schema import Document
from graph.chat_graph import build_graph
import global_retriever
from dotenv import load_dotenv
from chat_history import load_chat_history, save_chat_history  # Import your chat history functions

load_dotenv()

app = FastAPI()


# Function to load parsed documents
def load_or_parse_documents(screenshot_dir=SCREENSHOT_DIR):
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            texts = [doc["page_content"] for doc in json.load(f)]
    else:
        texts = []
    return texts


# Define an async chat function to handle the async `graph.ainvoke()`
async def chat_async(session_id: str, chat_history: List[BaseMessage]) -> str:
    # 1. Load parsed documents
    texts = load_or_parse_documents()
    all_chunks = [Document(page_content=text) for text in texts]

    # 2. Set up vectorstore
    embeddings = DialLabEmbeddings(
        model=os.getenv("DIAL_LAB_MODEL"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        base_url=os.getenv("DIAL_LAB_BASE_URL")
    )

    vectorstore = load_or_build_vectorstore(all_chunks, embeddings)

    retriever = DialLabRetriever(
        model=embeddings.model,
        api_key=embeddings.api_key,
        base_url=os.getenv("DIAL_LAB_BASE_URL"),
        faiss_index=vectorstore
    )

    # Set global retriever for generate_response
    global_retriever.retriever = retriever
    print("Global retriever set:", global_retriever.retriever)

    # 3. Build the graph and invoke it asynchronously using `ainvoke`
    graph = build_graph(
        session_id=session_id,
        faiss_index_path=FAISS_INDEX_PATH,
        model=embeddings.model,
        api_key=embeddings.api_key
    )

    state = {
        "messages": chat_history,
        "session_id": session_id,
        "faiss_index_path": FAISS_INDEX_PATH,
        "model": embeddings.model,
        "api_key": embeddings.api_key
    }

    print("State being passed into graph.ainvoke():", state)

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": session_id}})

    reply = result["messages"][-1].content if result.get("messages") else "No response generated."
    return reply


@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", str(uuid.uuid4()))  # Generate new if not provided

    # 1. Load the chat history
    chat_history = load_chat_history(session_id)

    # 2. Add the new user message
    chat_history.append(HumanMessage(content=user_message))

    # 3. Get the AI reply using full history
    reply = await chat_async(session_id, chat_history)

    # 4. Append AI response
    chat_history.append(AIMessage(content=reply))

    # 5. Save updated history
    save_chat_history(session_id, chat_history)

    return {"response": reply, "session_id": session_id}
