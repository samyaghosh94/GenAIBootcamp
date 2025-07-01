# test_mongo_embeddings.py

import os
import google.generativeai as genai
from vectorstore.mongo_vectorstore import save_embedding, get_all_embeddings
from dotenv import load_dotenv

load_dotenv()

def test_save_and_fetch():
    # Configure Gemini embeddings
    genai.configure(api_key=os.getenv("EMBEDDING_KEY"))

    sample_text = "This is a test document about HCM portal."

    # Embed the sample text
    response = genai.embed_content(
        model="models/text-embedding-004",  # Make sure this is the correct model
        content=sample_text,
        task_type="retrieval_document"  # or "semantic_similarity" depending on use
    )

    vector = response["embedding"]

    # Save to MongoDB
    doc_id = save_embedding(sample_text, vector, metadata={"source": "test"})
    print(f"✅ Saved document with ID: {doc_id}")

    # Fetch all embeddings back
    records = get_all_embeddings()
    assert any(r["_id"] == doc_id for r in records), "🛑 Saved document not found in MongoDB!"
    print("✅ Verified saved document in MongoDB. Total records:", len(records))

if __name__ == "__main__":
    test_save_and_fetch()
