# custom_diallab_retriever.py

import os
from typing import List
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langsmith import traceable
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings  # Your custom embeddings class

@traceable()
class DialLabRetriever:
    def __init__(self, model: str, api_key: str, base_url: str, faiss_index: FAISS):
        """
        Custom Retriever that uses DialLabEmbeddings for retrieving relevant documents.

        Args:
            model: The model identifier for your embeddings model.
            api_key: Your API key for authenticating the request.
            base_url: The base URL for your embedding service.
            faiss_index: FAISS index where the documents are stored.
        """
        self.embeddings = DialLabEmbeddings(model=model, api_key=api_key, base_url=base_url)
        self.faiss_index = faiss_index

    def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query text into the same vector space as the document embeddings.
        """
        return self.embeddings.embed_query(query)

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the relevant documents using FAISS and custom embeddings.

        Args:
            query: The user query.
            k: Number of documents to retrieve.

        Returns:
            List[Document]: A list of documents that are most relevant to the query.
        """
        query_embedding = self._embed_query(query)
        return self.faiss_index.similarity_search_by_vector(query_embedding, k=k)


# Optional test block for standalone testing
if __name__ == "__main__":
    load_dotenv()

    # Example documents
    texts = [
        "This is a test document.",
        "This is another test document.",
        "This is a third test document."
    ]
    documents = [Document(page_content=text) for text in texts]

    # Initialize embeddings
    embeddings = DialLabEmbeddings(
        model=os.getenv("DIAL_LAB_MODEL", "text-embedding-3-small-1"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        base_url=os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com")
    )

    # Create FAISS vector store
    faiss_vectorstore = FAISS.from_documents(documents, embeddings)

    # Use the custom retriever
    retriever = DialLabRetriever(
        model=embeddings.model,
        api_key=embeddings.api_key,
        base_url=embeddings.base_url,
        faiss_index=faiss_vectorstore
    )

    # Run a test query
    user_input = "What is a test document?"
    docs = retriever.get_relevant_documents(user_input)
    print(f"Relevant documents for the query '{user_input}':")
    for doc in docs:
        print(f" - {doc.page_content}")
