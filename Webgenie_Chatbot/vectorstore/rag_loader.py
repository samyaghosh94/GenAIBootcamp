import os
import json
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from vectorstore.mongo_vectorstore import clear_vectorstore, save_embedding, vector_similarity_search
from vectorstore.gemini_embeddings import GeminiEmbeddings
# from mongo_vectorstore import clear_vectorstore, save_embedding, vector_similarity_search
# from gemini_embeddings import GeminiEmbeddings
from config import QNA_PARSED_TEXT_PATH, DOCX_SOURCE_PATH
load_dotenv()

def load_rag_vectorstore(embeddings, docx_path: str):
    """
    Loads and saves embeddings into MongoDB vector store from a .docx file.
    """
    print(f"📄 Loading DOCX using UnstructuredLoader: {docx_path}")

    loader = UnstructuredLoader(
        file_path=docx_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        include_orig_elements=False
    )

    documents = loader.load()

    # Save parsed content (for debugging or reuse)
    os.makedirs(os.path.dirname(QNA_PARSED_TEXT_PATH), exist_ok=True)
    with open(QNA_PARSED_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page_content": doc.page_content} for doc in documents], f, indent=2, ensure_ascii=False)

    # Clear old embeddings first
    print("🧹 Clearing old embeddings from MongoDB vector store...")
    clear_vectorstore()

    # Save new embeddings to MongoDB
    print("🧠 Saving embeddings to MongoDB vector store...")
    for doc in documents:
        embedding = embeddings.embed_query(doc.page_content)
        save_embedding(doc.page_content, embedding)

    print("✅ Completed saving all embeddings to MongoDB.")
    return documents


if __name__ == '__main__':
    embeddings = GeminiEmbeddings(api_key=os.getenv("EMBEDDING_KEY"))
    docx_path = DOCX_SOURCE_PATH

    # Load and save embeddings
    docs = load_rag_vectorstore(embeddings, docx_path)

    # Run a test similarity search
    sample_query = "How to add a new employee record?"
    print(f"\n🔎 Running similarity search for query: {sample_query}")
    query_embedding = embeddings.embed_query(sample_query)
    results = vector_similarity_search(query_embedding, k=3)

    print("\n🎯 Search results:")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: {doc.page_content[:300]}...\n")
