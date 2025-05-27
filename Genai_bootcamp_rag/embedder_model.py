import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

def embeddings(split_docs: List[Document]) -> Chroma:
    """
    Create vector embeddings using OpenAI embeddings and Chroma for a set of documents.

    Args:
        split_docs (List[Document]): A list of smaller document chunks.

    Returns:
        Chroma: A Chroma vector store containing the embeddings.
    """
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        api_version=os.getenv("AZURE_API_VERSION")

    )
    return Chroma.from_documents(split_docs, embeddings)