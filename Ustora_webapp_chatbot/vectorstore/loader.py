import os
import json
import faiss
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore.in_memory import InMemoryDocstore
from config import FAISS_INDEX_PATH, DOCSTORE_PATH, QNA_PARSED_PATH

def load_or_build_vectorstore(embeddings):
    # Step 1: Load OmniParser (screenshot) documents
    if not os.path.exists(DOCSTORE_PATH):
        raise FileNotFoundError(f"Missing required file: {DOCSTORE_PATH}")
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        omni_docs = [Document(**doc) for doc in json.load(f)]

    # Step 2: Load QnA documents from JSON
    if not os.path.exists(QNA_PARSED_PATH):
        raise FileNotFoundError(f"Missing required file: {QNA_PARSED_PATH}")
    with open(QNA_PARSED_PATH, "r", encoding="utf-8") as f:
        qna_json = json.load(f)

    # Convert QnA dict to Document list
    qna_docs = [
        Document(
            page_content=f"Q: {question}\nA: {answer['answer'] if isinstance(answer['answer'], str) else json.dumps(answer['answer'])}"
        )
        for question, answer in qna_json.items()
    ]

    # Step 3: Combine all chunks
    all_docs = omni_docs + qna_docs

    # Step 4: If FAISS index already exists, load it
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Build docstore only from omni_docs (because that’s what's in documents.json)
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(omni_docs)})

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(omni_docs))}
        )
    else:
        print("⚙️ Building FAISS index from scratch...")
        vectorstore = FAISS.from_documents(all_docs, embeddings)

        # Save FAISS index only
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)

        # DO NOT overwrite documents.json here — it's already persisted

    return vectorstore
