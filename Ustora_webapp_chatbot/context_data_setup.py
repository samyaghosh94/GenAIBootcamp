import json
import os
import asyncio
from screenshot_capture import capture_screenshots_for_all_pages
from parsers.omniparser_client import parse_image_with_retries
from config import SCREENSHOT_DIR, DOCSTORE_PATH

def save_app_context(parsed_context):
    """
    Save the parsed context to a JSON file.
    """
    if parsed_context:
        os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump([{"page_content": text} for text in parsed_context], f)
    else:
        print("⚠️ No context to save. Parsed context is empty.")
        return

async def process_screenshots_and_store_context():
    """
    Capture screenshots (optional), parse them with OmniParser,
    and store the semantics in a context store.
    """
    # Uncomment if you want to re-capture screenshots each time
    print("📸 Capturing screenshots...")
    await capture_screenshots_for_all_pages()

    parsed_results = []

    for idx, filename in enumerate(sorted(os.listdir(SCREENSHOT_DIR))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            screenshot_path = os.path.join(SCREENSHOT_DIR, filename)
            print(f"\n🟡 Processing screenshot {idx + 1}: {filename}")

            result = await parse_image_with_retries(screenshot_path)

            if result:
                parsed_results.append(result if isinstance(result, str) else result.get("content", ""))
            else:
                print(f"❌ Failed to parse {filename} after retries")

    # Save all semantics to a JSON context store
    save_app_context(parsed_results)

async def main():
    await process_screenshots_and_store_context()

if __name__ == "__main__":
    asyncio.run(main())