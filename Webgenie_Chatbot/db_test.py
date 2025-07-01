from pymongo import MongoClient
from uuid import uuid4
import os
from dotenv import load_dotenv
load_dotenv()

# Use hardcoded URI or from environment
# MONGO_URI = "mongodb+srv://samyaghosh:yucDguD1JQN2DoxG@rag-db.hdjdbnj.mongodb.net/rag_db?retryWrites=true&w=majority&appName=RAG-DB"
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Step 1: Test connection
    client.admin.command('ping')
    print("✅ Connected successfully to MongoDB Atlas!")

    # Step 2: Test write access
    db = client["rag_db"]
    collection = db["test_write_access"]

    test_doc = {
        "_id": str(uuid4()),
        "test": "ping",
        "status": "testing write access"
    }

    insert_result = collection.insert_one(test_doc)
    print(f"✅ Successfully inserted test document with ID: {insert_result.inserted_id}")

    # # Optional: Cleanup test document
    # collection.delete_one({"_id": test_doc["_id"]})
    # print("🧹 Test document deleted — cleanup complete.")

except Exception as e:
    print("🛑 MongoDB connection or insert failed:", e)
