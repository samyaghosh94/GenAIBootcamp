# graph_nodes.py

import asyncio

import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from model.gemini_client import call_gemini_api  # Your async Gemini API call
from langchain.schema import Document
from langchain_core.messages.base import BaseMessage
from typing import List, TypedDict
import global_retriever

def count_tokens(messages: List[BaseMessage]) -> int:
    # Initialize token count
    token_count = 0
    for message in messages:
        # Assuming each message has a 'content' attribute
        token_count += len(message.content.split())  # Basic word count for tokens
    return token_count

# Define the updated ChatState with faiss_index, model, and api_key
class ChatState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    faiss_index_path: str
    model: str
    api_key: str


def generate_response(state: ChatState) -> ChatState:
    # 1. Get the last human message
    user_query = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content

    retriever = global_retriever.retriever
    # 🔒 SAFETY CHECK: Ensure global_retriever is initialized
    if retriever is None:
        response_text = "⚠️ Internal error: Retriever not initialized."
        state["messages"].append(AIMessage(content=response_text))
        return state

    # 2. Use FAISS index to fetch relevant context
    model = state["model"]
    api_key = state["api_key"]

    # Assuming faiss_index has a method to retrieve relevant documents based on the query
    try:
        # Modify this method based on how the FAISS index is structured in your application
        documents: List[Document] = global_retriever.retriever.get_relevant_documents(user_query)
    except Exception as e:
        response_text = f"⚠️ Error retrieving documents: {str(e)}"
        state["messages"].append(AIMessage(content=response_text))
        return state

    # 3. Prepare context from retrieved documents
    context = "\n".join([doc.page_content for doc in documents if doc.page_content.strip()])

    # Define the trimmer (with a max token count of, e.g., 100 tokens)
    trimmer = trim_messages(
        max_tokens=100,  # Max tokens allowed in the history
        strategy="last",  # You can choose "last" to trim from the end or "first" from the start
        token_counter=count_tokens,  # Function to count tokens (you can adjust as per your model)
        include_system=True,  # Whether to include system messages in the trimming logic
        allow_partial=False  # If True, allows partial trimming of messages
    )

    # Trim the chat history before adding the new response
    trimmed_history = trimmer.invoke(state["messages"])

    # --- Step 2: Prepare combined context ---
    # Chat history as text
    history_context = ""
    for msg in trimmed_history:
        if isinstance(msg, HumanMessage):
            history_context += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_context += f"Assistant: {msg.content}\n"

    # Final context to send to Gemini
    combined_context = f"""### Chat History
    {history_context}

    ### Relevant App Info
    {context}
    
    ### Current Question
    {user_query}
    """

    # Update the state with the trimmed chat history
    state["messages"] = trimmed_history

    # 4. Query Gemini model with the retrieved context
    if not context.strip():
        response_text = "⚠️ I couldn't find anything relevant in the app."
    else:
        try:
            # Call the Gemini API asynchronously
            print("Context getting passed to Gemini API:", combined_context)
            response_text = asyncio.run(call_gemini_api(user_query, combined_context))
        except Exception as e:
            response_text = f"⚠️ Error calling Gemini: {str(e)}"

    # 5. Append the AI response to the messages in state
    state["messages"].append(AIMessage(content=response_text))

    return state
