import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])

def create_embeddings_store(split_docs: List[Document]) -> Chroma:
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        api_version=os.getenv("AZURE_API_VERSION")
    )
    return Chroma.from_documents(split_docs, embeddings)

def retriever(query: str, vector_db: Chroma) -> List[Document]:
    return vector_db.as_retriever(search_kwargs={"k": 3}).invoke(query)

def main():
    pdf_path = r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Capstone Agents Project\data\faq.pdf"  # Ensure this file exists
    print("📄 Extracting PDF text...")
    text = extract_pdf_text(pdf_path)

    print("✂️ Chunking text...")
    split_docs = chunk_text(text)

    print("🔢 Creating embeddings and vector store...")
    vectordb = create_embeddings_store(split_docs)

    query = "Do you offer late checkout?"
    print(f"🔍 Query: {query}")
    results = retriever(query, vectordb)

    print("\n🧠 Top matching chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content.strip()}\n")

if __name__ == "__main__":
    main()
