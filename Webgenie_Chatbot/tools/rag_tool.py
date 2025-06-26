# rag_tool.py

from autogen_core.tools import FunctionTool
from typing import List
from langchain.schema import Document
from config import DOCX_SOURCE_PATH
from vectorstore.rag_loader import load_rag_vectorstore
from langchain_openai import AzureOpenAIEmbeddings
from vectorstore.gemini_embeddings import GeminiEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Load embeddings and vectorstore once
# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-small-1",
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     api_key=os.getenv("DIAL_LAB_KEY"),
#     api_version=os.getenv("AZURE_API_VERSION")
# )

embeddings = GeminiEmbeddings(api_key=os.getenv("EMBEDDING_KEY"))
print("📦 Loading vectorstore...")
vectorstore = load_rag_vectorstore(embeddings, DOCX_SOURCE_PATH)
print("✅ Vectorstore loaded.")

def retrieve_context(query: str) -> str:
    print(f"\n🔍 Received query: {query}")
    docs: List[Document] = vectorstore.similarity_search(query, k=3)
    print(f"📊 Retrieved {len(docs)} documents from FAISS index")
    if not docs:
        return "No relevant information found."
    for i, doc in enumerate(docs):
        print(f"\n📄 Match {i + 1}:\n{doc.page_content[:300]}...\n")
    return "\n\n".join([doc.page_content for doc in docs])

rag_tool = FunctionTool(
    name="retrieve_context",
    description="Retrieves relevant context from Ustora's FAQ and shopping database.",
    func=retrieve_context
)
