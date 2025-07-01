# gemini_embeddings.py
import google.generativeai as genai
from typing import List

class GeminiEmbeddings:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "models/gemini-embedding-exp-03-07"  # Use correct version from Google

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"  # or "semantic_similarity"
        )
        return response["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)  # 💡 This makes the object callable
