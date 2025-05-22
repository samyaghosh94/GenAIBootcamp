import json
import os
import faiss
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore.in_memory import InMemoryDocstore
from config import FAISS_INDEX_PATH, DOCSTORE_PATH, QNA_PARSED_TEXT_PATH, QNA_PARSED_PATH
from dotenv import load_dotenv

load_dotenv()


def load_qna_texts():
    if os.path.exists(QNA_PARSED_TEXT_PATH):
        with open(QNA_PARSED_TEXT_PATH, "r", encoding="utf-8") as f:
            qna_texts = [item["page_content"] for item in json.load(f)]
        print(f"✅ Loaded {len(qna_texts)} QnA texts for embedding.")
        return qna_texts
    return []


def load_screenshot_data():
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            screenshot_data = [Document(page_content=item["page_content"]) for item in json.load(f)]
        print(f"✅ Loaded {len(screenshot_data)} screenshot documents.")
        return screenshot_data
    return []


def load_or_build_vectorstore(embeddings):
    """Load or build a combined FAISS vectorstore for screenshots + QnA."""
    # Load both sets of documents
    screenshot_documents = load_screenshot_data()
    qna_texts = load_qna_texts()
    qna_documents = [Document(page_content=text) for text in qna_texts]
    all_documents = screenshot_documents + qna_documents

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCSTORE_PATH) and os.path.exists(QNA_PARSED_TEXT_PATH):
        print("🔄 Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Create a new docstore from combined documents
        docstore_dict = {str(i): doc for i, doc in enumerate(all_documents)}
        docstore = InMemoryDocstore(docstore_dict)

        index_to_docstore_id = {i: str(i) for i in range(len(all_documents))}

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        print("✅ Vector store loaded successfully.")
        return vectorstore

    # If no saved index, build from scratch
    print("⚙️ Building FAISS index from scratch...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)

    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in all_documents], f, ensure_ascii=False, indent=4)

    print("✅ Vector store created and saved.")
    return vectorstore
