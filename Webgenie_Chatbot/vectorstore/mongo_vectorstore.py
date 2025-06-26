# mongo_vectorstore.py

from pymongo import MongoClient
from typing import List
import os
from uuid import uuid4

MONGO_URI = "mongodb+srv://samyaghosh:yucDguD1JQN2DoxG@rag-db.hdjdbnj.mongodb.net/rag_db?retryWrites=true&w=majority&appName=RAG-DB"

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["rag_db"]
collection = db["test_rag_vectorstore"]


def save_embedding(doc_content: str, embedding: List[float], metadata: dict = None):
    # Step 1: Test connection
    client.admin.command('ping')
    print("✅ Connected successfully to MongoDB Atlas!")

    metadata = metadata or {}
    doc_id = str(uuid4())
    vectordoc = {
        "_id": doc_id,
        "content": doc_content,
        "embedding": embedding,
        "metadata": metadata
    }
    vector_insert = collection.insert_one(vectordoc)
    print(f"✅ Successfully inserted test document with ID: {vector_insert.inserted_id}")
    return doc_id


def get_all_embeddings():
    return list(collection.find())
