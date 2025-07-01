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


def clear_vectorstore():
    """Delete all documents in the vector store (do this once before re-embedding)."""
    client.admin.command('ping')
    delete_result = collection.delete_many({})
    print(f"🧹 Deleted {delete_result.deleted_count} documents from the collection.")


def save_embedding(doc_content: str, embedding: List[float], metadata: Optional[dict] = None):
    """Insert a single document and its embedding into MongoDB."""
    metadata = metadata or {}
    doc_id = str(uuid4())
    vectordoc = {
        "_id": doc_id,
        "content": doc_content,
        "embedding": embedding,
        "metadata": metadata
    }
    result = collection.insert_one(vectordoc)
    print(f"✅ Inserted document with ID: {result.inserted_id}")
    return doc_id


def get_all_embeddings():
    return list(collection.find())


def vector_similarity_search(query_embedding: List[float], k: int = 5) -> List[Document]:
    """Perform semantic similarity search using MongoDB Vector Search."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
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

    results = collection.aggregate(pipeline)
    return [
        Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
        for doc in results
    ]
