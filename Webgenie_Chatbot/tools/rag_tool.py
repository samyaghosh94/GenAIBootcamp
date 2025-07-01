from autogen_core.tools import FunctionTool
from typing import List
from langchain.schema import Document
from config import DOCX_SOURCE_PATH
from vectorstore.rag_loader import load_rag_vectorstore
from vectorstore.mongo_vectorstore import vector_similarity_search  # Your MongoDB search function
from vectorstore.gemini_embeddings import GeminiEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = GeminiEmbeddings(api_key=os.getenv("GENAI_PLUS"))
print("📦 Loading vectorstore...")
# Load documents but no FAISS index now
documents = load_rag_vectorstore(embeddings, DOCX_SOURCE_PATH)
print("✅ Vectorstore loaded.")

def retrieve_context(query: str) -> str:
    print(f"\n🔍 Received query: {query}")
    query_embedding = embeddings.embed_query(query)

    # Run vector similarity search against MongoDB vector store
    results = vector_similarity_search(query_embedding, k=3)
    print(f"📊 Retrieved {len(results)} documents from MongoDB vector store")

    if not results:
        return "No relevant information found."

    # Convert your result dicts to Documents if needed
    docs = results

    for i, doc in enumerate(docs):
        print(f"\n📄 Match {i + 1}:\n{doc.page_content[:300]}...\n")

    return "\n\n".join([doc.page_content for doc in docs])


rag_tool = FunctionTool(
    name="retrieve_context",
    description="Retrieves relevant context from Ustora's FAQ and shopping database.",
    func=retrieve_context
)
