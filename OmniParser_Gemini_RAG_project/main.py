# main.py

import os
import asyncio
import time
from screenshot_capture import capture_screenshots_for_all_pages
from omniparser_client import parse_image_with_retries
from gemini_client import call_gemini_api
from context_store import save_context, load_context
from config import SCREENSHOT_DIR

# Set your user prompt here
USER_PROMPT = "Where is my cart?"


async def process_screenshots_and_store_context():
    """
    Capture screenshots (optional), parse them with OmniParser,
    and store the semantics in a context store.
    """
    # Uncomment if you want to re-capture screenshots each time
    print("📸 Capturing screenshots...")
    # await capture_screenshots_for_all_pages()

    parsed_context = {}

    for idx, filename in enumerate(sorted(os.listdir(SCREENSHOT_DIR))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            screenshot_path = os.path.join(SCREENSHOT_DIR, filename)
            print(f"\n🟡 Processing screenshot {idx + 1}: {filename}")

            start_time = time.time()
            result = await parse_image_with_retries(screenshot_path)
            end_time = time.time()

            if result:
                parsed_context[filename] = result
                print(f"✅ Parsed {filename} (Time: {end_time - start_time:.2f}s)")
            else:
                parsed_context[filename] = ""
                print(f"❌ Failed to parse {filename} after retries")

    # Save all semantics to a JSON context store
    save_context(parsed_context)


async def run_gemini_with_context():
    """
    Load the saved context and use it to call Gemini API with the user prompt.
    """
    context = load_context()

    if not context.strip():
        print("⚠️ No context loaded. Did parsing fail or context file not exist?")
        return

    print("\n🧠 Sending prompt and context to Gemini...")
    response = await call_gemini_api(USER_PROMPT, context)

    if response:
        print("\n💬 Gemini Response:\n")
        print(response)
    else:
        print("❌ Failed to get a response from Gemini.")


async def main():
    await process_screenshots_and_store_context()
    await run_gemini_with_context()


if __name__ == "__main__":
    asyncio.run(main())
