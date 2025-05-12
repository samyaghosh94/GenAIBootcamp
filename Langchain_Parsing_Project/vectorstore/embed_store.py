import os
from dotenv import load_dotenv
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

from vectorstore.custom_diallab_embeddings import DialLabEmbeddings

# Load environment variables from .env
load_dotenv()

# Fallback env vars
DIAL_LAB_KEY = os.getenv("DIAL_LAB_KEY")
DIAL_LAB_BASE_URL = os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com")
DIAL_LAB_MODEL = os.getenv("DIAL_LAB_MODEL", "text-embedding-3-small-1")


def create_vectorstore(
    parsed_texts: List[str],
    embeddings: Optional[DialLabEmbeddings] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    # Step 1: Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 2: Split texts into Document chunks
    all_chunks = []
    for text in parsed_texts:
        chunks = splitter.split_text(text)
        all_chunks.extend([Document(page_content=chunk) for chunk in chunks])

    # Step 3: Use passed embeddings or fallback to default
    if embeddings is None:
        embeddings = DialLabEmbeddings(
            model=DIAL_LAB_MODEL,
            api_key=DIAL_LAB_KEY,
            base_url=DIAL_LAB_BASE_URL
        )

    # Optional: print chunks + partial embeddings
    for doc in all_chunks:
        print(f"\n🔹 Chunk:\n{doc.page_content}")
        try:
            emb = embeddings.embed_query(doc.page_content)
            print(f"🧠 Embedding (first 5 values): {emb[:5]}")
        except Exception as e:
            print(f"❌ Error embedding chunk: {e}")
        print("-" * 50)

    # Step 4: Create FAISS vector store
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    return vectorstore
