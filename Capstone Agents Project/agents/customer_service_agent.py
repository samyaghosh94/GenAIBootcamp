from core.hotel_state import HotelState
from langchain.chat_models import init_chat_model
import fitz  # PyMuPDF
from typing import Any, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = init_chat_model(
    model="gpt-4o-mini-2024-07-18",
    model_provider="azure_openai",
    api_key=os.getenv("DIAL_LAB_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> list[Document]:
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

def customer_service_agent(state: HotelState) -> dict:
    print("\n🛎️ Running Customer Service Agent...")

    customer_name = state["booking"].get("customer_name", "Guest")
    inquiry = state["request"].get("inquiry", "No inquiry provided")

    print(f"📨 Inquiry from {customer_name}: {inquiry}")

    # Step 1: Extract and chunk the PDF
    faq_text = extract_pdf_text(r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Capstone Agents Project\data\faq.pdf")
    split_docs = chunk_text(faq_text)

    # Step 2: Build vector store and retrieve top relevant chunks
    vector_db = create_embeddings_store(split_docs)
    results = retriever(inquiry, vector_db)
    chunks = [doc.page_content for doc in results]

    if not chunks:
        response = "Sorry, I couldn't find any relevant information in the FAQs."
    else:
        # Step 3: Create prompt using retrieved chunks
        context = "\n---\n".join(chunks)
        prompt = f"""
You are a helpful hotel assistant. Use the following FAQ information to answer the customer's question.

FAQ Context:
{context}

Customer Inquiry:
{inquiry}

Answer:
"""

        # Step 4: Call Gemini (or your chat model)
        response = chat_model.predict(prompt)

    customer_service_info = {
        "inquiry": inquiry,
        "response": response
    }

    print(f"\n✅ Customer service response:\n{response}\n")

    # Return new updated state dict with customer_service info updated
    return {**state, "customer_service": customer_service_info}


