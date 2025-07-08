from pymongo import MongoClient
from typing import List, Optional
from uuid import uuid4
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag_db"
COLLECTION_NAME = "test_rag_vectorstore"
VECTOR_INDEX_NAME = "vector_index"  # This must match the name of your vector index in Atlas

# Connect to MongoDB
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def collection_has_data(collection_name_override: Optional[str] = None) -> bool:
    """
    Checks if the specified collection already contains documents.
    Returns True if data exists, False otherwise.
    """
    target_collection = db[collection_name_override or COLLECTION_NAME]
    count = target_collection.estimated_document_count()
    print(f"🔎 Collection '{target_collection.name}' has {count} documents.")
    return count > 0



def clear_vectorstore(collection_name_override: Optional[str] = None):
    """Delete all documents in the vector store (e.g., before re-embedding)."""
    client.admin.command('ping')
    target_collection = db[collection_name_override or COLLECTION_NAME]
    delete_result = target_collection.delete_many({})
    print(f"🧹 Deleted {delete_result.deleted_count} documents from collection '{target_collection.name}'")



def save_embedding(doc_content: str, embedding: List[float], metadata: Optional[dict] = None, collection_name_override: Optional[str] = None):
    metadata = metadata or {}
    doc_id = str(uuid4())
    vectordoc = {
        "_id": doc_id,
        "content": doc_content,
        "embedding": embedding,
        "metadata": metadata
    }
    target_collection = db[collection_name_override or COLLECTION_NAME]
    result = target_collection.insert_one(vectordoc)
    print(f"✅ Inserted into '{target_collection.name}' with ID: {result.inserted_id}")
    return doc_id


def get_all_embeddings():
    return list(collection.find())


def vector_similarity_search(
    query_embedding: List[float],
    k: int = 5,
    collection_name_override: Optional[str] = None,
    index_name_override: Optional[str] = None
) -> List[Document]:
    target_collection = db[collection_name_override or COLLECTION_NAME]
    index_name = index_name_override or VECTOR_INDEX_NAME

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "content": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = target_collection.aggregate(pipeline)
    return [
        Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
        for doc in results
    ]


