import os
import json
import asyncio

from dotenv import load_dotenv
from config import SCREENSHOT_DIR, DOCSTORE_PATH
from screenshot_capture import capture_screenshots_for_all_pages
from parsers.omniparser_client import parse_image_with_retries
from model.gemini_client import call_gemini_api
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings
from vectorstore.custom_diallab_retriever import DialLabRetriever
from vectorstore.loader import load_or_build_vectorstore

from langchain.schema import Document

# Load environment variables
load_dotenv()

# DOCSTORE_PATH = "storage/documents.json"


def parse_all_screenshots(screenshot_dir=SCREENSHOT_DIR):
    parsed_results = []

    for file in os.listdir(screenshot_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(screenshot_dir, file)
            print(f"⏳ Submitting {file} to OmniParser...")

            result = parse_image_with_retries(full_path)
            if result:
                parsed_results.append(result if isinstance(result, str) else result.get("content", ""))
            else:
                print(f"❌ Failed to parse {file}")

    return parsed_results


def load_or_parse_documents(screenshot_dir=SCREENSHOT_DIR):
    """
    Load previously parsed documents from disk, or parse screenshots if not available.
    """
    if os.path.exists(DOCSTORE_PATH):
        print("📄 Loading parsed documents from disk...")
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            texts = [doc["page_content"] for doc in json.load(f)]
    else:
        print("🧠 Parsing screenshots with OmniParser...")
        texts = parse_all_screenshots(screenshot_dir)
        os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump([{"page_content": text} for text in texts], f)
        print(f"✅ Parsed and saved {len(texts)} documents.\n")

    return texts


def run_pipeline():
    # Step 1: Optional - Capture new screenshots
    # print("📸 Capturing screenshots from web app...")
    # capture_screenshots_for_all_pages()
    # print("✅ Screenshots captured.\n")

    # Step 2–3: Load or parse screenshot documents
    parsed_texts = load_or_parse_documents()
    if not parsed_texts:
        print("⚠️ No parsed documents available. Exiting.")
        return

    # Step 4: Convert to LangChain Documents
    all_chunks = [Document(page_content=text) for text in parsed_texts]

    # Step 5: Initialize DialLab Embeddings
    embeddings = DialLabEmbeddings(
        model=os.getenv("DIAL_LAB_MODEL", "text-embedding-3-small-1"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        base_url=os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com")
    )

    # Step 6: Load or build FAISS vectorstore
    print("📦 Preparing FAISS vectorstore...")
    vectorstore = load_or_build_vectorstore(all_chunks, embeddings)
    print("✅ Vectorstore ready.\n")

    # Step 7: Accept user query
    print("🤖 Ask your question about the app:")
    user_input = input("> ").strip()
    if not user_input:
        print("⚠️ Empty input received. Exiting.")
        return

    # Step 8: Retrieve relevant documents
    retriever = DialLabRetriever(
        model=embeddings.model,
        api_key=embeddings.api_key,
        base_url=os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com"),
        faiss_index=vectorstore
    )

    print("🔎 Retrieving relevant documents...")
    docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])

    if not context.strip():
        print("⚠️ No relevant context found. Skipping Gemini call.")
        return

    # Step 9: Call Gemini API
    print("📡 Calling Gemini API...")
    model_response = asyncio.run(call_gemini_api(user_input, context))

    # Step 10: Display response
    print("\n💬 Gemini Response:")
    print(model_response)


if __name__ == "__main__":
    run_pipeline()
