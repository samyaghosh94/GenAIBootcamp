# rag_loader.py

import os
import json
import time

from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_unstructured import UnstructuredLoader
import faiss
from langchain_openai import AzureOpenAIEmbeddings
# from config import FAISS_INDEX_PATH, QNA_PARSED_TEXT_PATH
from dotenv import load_dotenv
from vectorstore.mongo_vectorstore import save_embedding
# from mongo_vectorstore import save_embedding
# from gemini_embeddings import GeminiEmbeddings
from vectorstore.gemini_embeddings import GeminiEmbeddings

load_dotenv()


def load_rag_vectorstore(embeddings, docx_path: str) -> FAISS:
    """
    Loads or builds a FAISS vectorstore from a .docx file using UnstructuredLoader.
    """
    FAISS_INDEX_PATH = r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Webgenie_Chatbot\storage\faiss_index.index"
    QNA_PARSED_TEXT_PATH = r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Webgenie_Chatbot\storage\qna_texts.json"
    print(f"📄 Loading DOCX using UnstructuredLoader: {docx_path}")

    loader = UnstructuredLoader(
        file_path=docx_path,
        strategy="hi_res",               # More accurate extraction
        chunking_strategy="by_title",       # Group into coherent sections
        include_orig_elements=False      # Skip raw elements
    )

    documents = loader.load()

    # Save parsed content (for debugging or reuse)
    os.makedirs(os.path.dirname(QNA_PARSED_TEXT_PATH), exist_ok=True)
    with open(QNA_PARSED_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page_content": doc.page_content} for doc in documents], f, indent=2, ensure_ascii=False)

    # ✅ NEW: Store each document’s embedding in MongoDB
    print("🧠 Saving embeddings to MongoDB...")
    for doc in documents:
        embedding = embeddings.embed_query(doc.page_content)
        save_embedding(doc.page_content, embedding)

    # Create or load FAISS index
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
        docstore = InMemoryDocstore(docstore_dict)
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    print("⚙️ Building new FAISS index...")
    time.sleep(120)  # Simulate long processing time for index creation
    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)

    print("✅ Vector store created and saved.")
    return vectorstore


if __name__ == '__main__':
    # embeddings = AzureOpenAIEmbeddings(
    #     model="text-embedding-3-small-1",
    #     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    #     api_key=os.getenv("DIAL_LAB_KEY"),
    #     api_version=os.getenv("AZURE_API_VERSION")
    # )
    embeddings = GeminiEmbeddings(api_key=os.getenv("EMBEDDING_KEY"))
    vectorstore = load_rag_vectorstore(embeddings,
                                       docx_path=r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Webgenie_Chatbot\storage\HCM_BackOffice_User_Help_Guide_Enhanced.docx")
