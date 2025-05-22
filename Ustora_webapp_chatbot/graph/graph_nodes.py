# graph_nodes.py

import asyncio
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langsmith import traceable
from model.gemini_client import call_gemini_api  # Your async Gemini API call
from langchain.schema import Document
from langchain_core.messages.base import BaseMessage
from typing import List, TypedDict
import global_retriever
from transformers import AutoTokenizer

from vertexai.preview.language_models import TextGenerationModel

def count_tokens(messages: List[BaseMessage]) -> int:
    # Initialize token count
    token_count = 0
    for message in messages:
        # Assuming each message has a 'content' attribute
        token_count += len(message.content.split())  # Basic word count for tokens
    return token_count

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=count_tokens,
    include_system=True,
    allow_partial=True,
)

class ChatState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    faiss_index_path: str
    model: str
    api_key: str

async def generate_response(state: ChatState) -> ChatState:
    user_query = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content

    retriever = global_retriever.retriever
    if retriever is None:
        response_text = "⚠️ Internal error: Retriever not initialized."
        state["messages"].append(AIMessage(content=response_text))
        return state

    try:
        documents: List[Document] = retriever.get_relevant_documents(user_query)
    except Exception as e:
        response_text = f"⚠️ Error retrieving documents: {str(e)}"
        state["messages"].append(AIMessage(content=response_text))
        return state

    context = "\n".join([doc.page_content for doc in documents if doc.page_content.strip()])
    trimmed_history = trimmer.invoke(state["messages"])

    history_context = ""
    for msg in trimmed_history:
        if isinstance(msg, HumanMessage):
            history_context += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_context += f"Assistant: {msg.content}\n"

    combined_context = f"""### Chat History\n{history_context}\n### Relevant App Info\n{context}\n### Current Question\n{user_query}\n"""
    print("Context passed to Gemini API:", combined_context)
    state["messages"] = trimmed_history

    if not context.strip():
        response_text = "⚠️ I couldn't find anything relevant in the app."
    else:
        try:
            response_text = await call_gemini_api(user_query, combined_context)
        except Exception as e:
            response_text = f"⚠️ Error calling Gemini: {str(e)}"

    state["messages"].append(AIMessage(content=response_text))
    return state
