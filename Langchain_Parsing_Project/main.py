import asyncio
import os
import time

from dotenv import load_dotenv
from config import SCREENSHOT_DIR
from screenshot_capture import capture_screenshots_for_all_pages
from parsers.omniparser_client import parse_image_with_retries
from vectorstore.embed_store import create_vectorstore
from model.gemini_client import call_gemini_api
from vectorstore.custom_diallab_retriever import DialLabRetriever
from vectorstore.custom_diallab_embeddings import DialLabEmbeddings

# Load environment variables
load_dotenv()


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


def run_pipeline():
    # Step 1: Capture screenshots
    print("📸 Capturing screenshots from web app...")
    # capture_screenshots_for_all_pages()
    print("✅ Screenshots captured.\n")

    # Step 2: Parse screenshots
    print("🧠 Parsing screenshots with OmniParser...")
    parsed_texts = parse_all_screenshots()
    print(f"✅ Parsed {len(parsed_texts)} screenshot(s).\n")

    # Step 3: Initialize embeddings once
    embeddings = DialLabEmbeddings(
        model=os.getenv("DIAL_LAB_MODEL", "text-embedding-3-small-1"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        base_url=os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com")
    )

    # Step 4: Create vector store using shared embeddings
    print("📦 Creating vector store...")
    vectorstore = create_vectorstore(parsed_texts, embeddings=embeddings)
    print("✅ Vector store created.\n")

    # Step 5: Accept user input
    print("🤖 Ask your question about the app:")
    user_input = input("> ").strip()

    if not user_input:
        print("⚠️ Empty input received. Exiting.")
        return

    # Step 6: Initialize custom retriever
    retriever = DialLabRetriever(
        model=embeddings.model,
        api_key=embeddings.api_key,
        base_url=os.getenv("DIAL_LAB_BASE_URL", "https://ai-proxy.lab.epam.com"),
        faiss_index=vectorstore
    )

    # Step 7: Retrieve documents
    print("🔎 Retrieving relevant documents using DialLabRetriever...")
    docs = retriever.get_relevant_documents(user_input)
    semantics = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])

    if not semantics.strip():
        print("⚠️ No relevant context found for the input. Skipping Gemini call.")
        return

    # Step 8: Call Gemini API
    print("📡 Calling Gemini API...")
    response = asyncio.run(call_gemini_api(user_input, semantics))

    # Step 9: Output response
    print("\n💬 Gemini Response:")
    print(response)


if __name__ == "__main__":
    run_pipeline()
