import json
import os
import uuid
from typing import List

from fastapi import FastAPI, Request, HTTPException
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from config import FAISS_INDEX_PATH, SCREENSHOT_DIR, DOCSTORE_PATH, QNA_PARSED_TEXT_PATH
from vectorstore.loader import load_or_build_vectorstore
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings
from vectorstore.custom_diallab_retriever import DialLabRetriever
from langchain.schema import Document
from graph.chat_graph import build_graph
import global_retriever
from dotenv import load_dotenv
from chat_history import load_chat_history, save_chat_history  # Import your chat history functions
from langsmith import traceable

load_dotenv()

app = FastAPI()

# Function to load parsed documents (screenshots)
def load_or_parse_documents():
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            texts = [doc["page_content"] for doc in json.load(f)]
    else:
        texts = []
    return texts

# Function to load QnA documents (directly from the pre-parsed qna_texts.json)
def load_qna_documents():
    if os.path.exists(QNA_PARSED_TEXT_PATH):
        with open(QNA_PARSED_TEXT_PATH, "r", encoding="utf-8") as f:
            qna_data = json.load(f)

        qna_texts = [item["page_content"] for item in qna_data]
        return qna_texts
    else:
        print("⚠️ QnA data not found!")
        return []


# Define an async chat function to handle the async `graph.ainvoke()`
@traceable()
async def chat_async(session_id: str, chat_history: List[BaseMessage]) -> str:
    # 1. Set up vectorstore embeddings
    embeddings = DialLabEmbeddings(
        model=os.getenv("DIAL_LAB_MODEL"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        base_url=os.getenv("DIAL_LAB_BASE_URL")
    )

    # 2. Load or build vectorstore from saved disk data
    vectorstore = load_or_build_vectorstore(embeddings)

    # 3. Create retriever
    retriever = DialLabRetriever(
        model=embeddings.model,
        api_key=embeddings.api_key,
        base_url=os.getenv("DIAL_LAB_BASE_URL"),
        faiss_index=vectorstore
    )

    global_retriever.retriever = retriever
    print("Global retriever set.")

    # 4. Build and run LangGraph
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
    try:
        reply = await chat_async(session_id, chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating reply: {str(e)}")

    # 4. Append AI response
    chat_history.append(AIMessage(content=reply))

    # 5. Save updated history
    save_chat_history(session_id, chat_history)

    return {"response": reply, "session_id": session_id}
