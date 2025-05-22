# custom_diallab_embeddings.py

import os
import requests
from typing import List

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langsmith import traceable

@traceable
class DialLabEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = f"{base_url}/openai/deployments/{self.model}/embeddings"
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "input": texts
        }
        response = requests.post(self.endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return [item["embedding"] for item in response.json()["data"]]
        else:
            raise Exception(f"Embedding API Error: {response.status_code} - {response.text}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

if __name__ == '__main__':
    load_dotenv()
    emb = DialLabEmbeddings(model="text-embedding-3-small-1", api_key=os.getenv("DIAL_LAB_KEY"),
                            base_url="https://ai-proxy.lab.epam.com")
    print(emb.embed_query("Test this embedding."))
