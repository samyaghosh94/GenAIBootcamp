from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore.in_memory import InMemoryDocstore
from config import FAISS_INDEX_PATH, DOCSTORE_PATH

import faiss
import json
import os

# FAISS_INDEX_PATH = "storage/faiss_index.index"
# DOCSTORE_PATH = "storage/documents.json"

def load_or_build_vectorstore(all_chunks, embeddings):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCSTORE_PATH):
        print("🔄 Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            documents = [Document(**doc) for doc in json.load(f)]

        # Build docstore from saved documents using a dict of {doc_id: Document}
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

        # Create a vectorstore object
        vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={i: str(i) for i in range(len(documents))})
    else:
        print("⚙️ Building FAISS index from scratch...")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)

        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump([doc.dict() for doc in all_chunks], f)

    return vectorstore
