from autogen_core.tools import FunctionTool
from pymongo import MongoClient
import os

mongo_uri = "mongodb+srv://samyaghosh:yucDguD1JQN2DoxG@rag-db.hdjdbnj.mongodb.net/rag_db?retryWrites=true&w=majority&appName=RAG-DB"
client = MongoClient(mongo_uri)
db = client.get_database("feedback_db")
collection = db.get_collection("user_feedback")

def save_feedback(data: dict):
    session_id = data.get("session_id", "unknown")
    feedback = data.get("feedback") or data.get("content")  # Fallback to 'content' for plain messages

    if not feedback:
        raise ValueError("Missing feedback content.")

    document = {
        "session_id": session_id,
        "feedback": feedback,
        "logs": data.get("logs", None),  # optional
    }
    collection.insert_one(document)
    return "Feedback saved successfully."

save_feedback_tool = FunctionTool(
    name="save_feedback_tool",
    func=save_feedback,
    description="Save structured user feedback to MongoDB. Required: 'feedback'. Optional: 'logs'."
)
