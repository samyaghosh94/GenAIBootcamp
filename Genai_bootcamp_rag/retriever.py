from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langsmith import traceable


def retriever(query: str, vector_db: Chroma) -> List[Document]:
    """
    Perform a similarity search query in the FAISS vector database.

    Args:
        query (str): User query string.
        vector_db (FAISS): The FAISS vector store containing document embeddings.

    Returns:
        List[Document]: A list of Document objects most similar to the query.
    """
    retriever = vector_db.as_retriever()
    return retriever.invoke(query)