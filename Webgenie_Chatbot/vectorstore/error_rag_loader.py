# error_rag_loader.py

import os
import json
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from vectorstore.gemini_embeddings import GeminiEmbeddings
from vectorstore.mongo_vectorstore import save_embedding, clear_vectorstore, vector_similarity_search, collection_has_data
from config import ERROR_DOCX_PATH, ERROR_PARSED_TEXT_PATH
# from mongo_vectorstore import clear_vectorstore, save_embedding, vector_similarity_search
# from gemini_embeddings import GeminiEmbeddings

load_dotenv()

# ERROR_DOCX_PATH = r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Webgenie_Chatbot\storage\HCM-Portal-TroubleshootingGuid.docx"
# ERROR_PARSED_TEXT_PATH = r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Webgenie_Chatbot\storage\error_parsed_texts.json"


def load_error_vectorstore(embeddings, docx_path: str):
    print(f"🛠 Loading Troubleshooting DOCX: {docx_path}")

    loader = UnstructuredLoader(
        file_path=docx_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        include_orig_elements=False
    )

    documents = loader.load()

    # Optional: Save parsed version for inspection
    os.makedirs(os.path.dirname(ERROR_PARSED_TEXT_PATH), exist_ok=True)
    with open(ERROR_PARSED_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page_content": doc.page_content} for doc in documents], f, indent=2, ensure_ascii=False)

    # ✅ Check for existing data
    collection_name = "error_vectorstore"
    if collection_has_data(collection_name_override=collection_name):
          print(f"ℹ️ Skipping re-embedding — data already exists in {collection_name}.")
          return documents

    # Clear old embeddings from troubleshooting collection
    print("🧹 Clearing previous error embeddings...")
    clear_vectorstore(collection_name_override="error_vectorstore")

    print("🧠 Saving new troubleshooting embeddings...")
    for doc in documents:
        embedding = embeddings.embed_query(doc.page_content)
        save_embedding(
            doc.page_content,
            embedding,
            metadata={"source": "troubleshooting"},
            collection_name_override="error_vectorstore"
        )

    print("✅ Troubleshooting vectorstore updated.")
    return documents

if __name__ == '__main__':
    embeddings = GeminiEmbeddings(api_key=os.getenv("EMBEDDING_KEY"))
    docx_path = ERROR_DOCX_PATH

    # Load and save embeddings
    docs = load_error_vectorstore(embeddings, docx_path)

    # Run a test similarity search
    sample_query = "How to add a new employee record?"
    print(f"\n🔎 Running similarity search for query: {sample_query}")
    query_embedding = embeddings.embed_query(sample_query)
    results = vector_similarity_search(query_embedding, k=3, collection_name_override="error_vectorstore")

    print("\n🎯 Search results:")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: {doc.page_content[:300]}...\n")