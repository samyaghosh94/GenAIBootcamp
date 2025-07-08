# error_rag_tool.py

from vectorstore.gemini_embeddings import GeminiEmbeddings
from vectorstore.mongo_vectorstore import vector_similarity_search
from langchain.schema import Document
import os
from dotenv import load_dotenv
from autogen_core.tools import FunctionTool
from vectorstore.error_rag_loader import load_error_vectorstore
from config import ERROR_DOCX_PATH, ERROR_PARSED_TEXT_PATH

load_dotenv()

embeddings = GeminiEmbeddings(api_key=os.getenv("GENAI_PLUS"))
print("📦 Loading vectorstore...")
# Load documents but no FAISS index now
documents = load_error_vectorstore(embeddings, ERROR_DOCX_PATH)
print("✅ Vectorstore loaded.")

def retrieve_troubleshooting_context(query: str) -> str:
    print(f"🔧 Searching troubleshooting docs for query: {query}")
    query_embedding = embeddings.embed_query(query)

    results = vector_similarity_search(query_embedding,
                                       k=3,
                                       collection_name_override="error_vectorstore",
                                       index_name_override="vector_index_error")
    if not results:
        return "No known troubleshooting steps found."

    for i, doc in enumerate(results):
        print(f"\n📄 Match {i + 1}:\n{doc.page_content[:300]}...\n")

    return "\n\n".join([doc.page_content for doc in results])

error_diagnosis_tool = FunctionTool(
    name="retrieve_troubleshooting_context",
    description="Retrieves known troubleshooting steps based on error reports or issue descriptions.",
    func=retrieve_troubleshooting_context
)