from autogen_core.tools import FunctionTool
from typing import List
from langchain.schema import Document

# Suppose these are already defined/imported in your RAG setup
from vectorstore.rag_loader import load_rag_vectorstore
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from langchain.chat_models import init_chat_model

from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro" or "gemini-2.0"
    temperature=0.2,
    max_tokens=1024,
    google_api_key=os.getenv("OPENAI_API_KEY")
)

# Load vector store once during startup
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small-1",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("DIAL_LAB_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)
vectorstore = load_rag_vectorstore(embeddings)

def answer_with_retrieved_context(query: str) -> str:
    docs: List[Document] = vectorstore.similarity_search(query, k=5)
    chunks = [doc.page_content for doc in docs]

    if not chunks:
        return  "I don't know."

    context = "\n---\n".join(chunks)
    prompt = f"""
You are a helpful assistant for the Ustora shopping application.
Use only the following context to answer the user's question.

Context:
{context}

User Question:
{query}

Answer:"""

    # ✅ Gemini response
    response = chat_model.predict(prompt)
    return str(response)


rag_tool = FunctionTool(
    name="answer_with_retrieved_context",
    description="Retrieves information from FAQs and answers user shopping queries using retrieved context.",
    func=answer_with_retrieved_context
)