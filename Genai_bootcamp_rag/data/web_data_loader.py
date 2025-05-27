from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_webpage(url: str) -> List:
    """
    Load a webpage and process it into a list of Document objects.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def text_splitting(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (List[Document]): The list of documents to split.

    Returns:
        List[Document]: A list of smaller document chunks.
    """
    # Split documents recursively and create smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    web_documents = text_splitter.split_documents(docs)
    return web_documents