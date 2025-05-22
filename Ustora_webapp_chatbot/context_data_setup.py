import json
import os
import asyncio
import re

from screenshot_capture import capture_screenshots_for_all_pages
from parsers.omniparser_client import parse_image_with_retries
from config import SCREENSHOT_DIR, DOCSTORE_PATH, QNA_DOC_PATH, QNA_PARSED_TEXT_PATH
from dotenv import load_dotenv

load_dotenv()


def save_app_context(parsed_context):
    """Save the parsed screenshot text context to a JSON file."""
    if parsed_context:
        os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump([{"page_content": text} for text in parsed_context], f, ensure_ascii=False, indent=4)
        print(f"✅ Screenshot context saved to {DOCSTORE_PATH}")
    else:
        print("⚠️ No screenshot context to save.")


def save_qna_texts(qna_texts):
    """Save flattened QnA texts (for embedding) to a JSON file."""
    if qna_texts:
        os.makedirs(os.path.dirname(QNA_PARSED_TEXT_PATH), exist_ok=True)
        with open(QNA_PARSED_TEXT_PATH, "w", encoding="utf-8") as f:
            json.dump([{"page_content": q} for q in qna_texts], f, ensure_ascii=False, indent=4)
        print(f"✅ Flattened QnA texts saved to {QNA_PARSED_TEXT_PATH}")
    else:
        print("⚠️ No QnA texts to save.")


def parse_qna_txt_file(file_path: str) -> list:
    """Parse a plain text QnA file into flattened QnA text chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    qna_texts = []
    current_question = None
    current_answer_lines = []
    options = {}

    for line in lines:
        if not line:
            continue

        # Match a question
        question_match = re.match(r"Q\d+[:：]\s*(.*)", line)
        if question_match:
            # Save previous
            if current_question:
                if options:
                    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
                    qna_texts.append(f"Q: {current_question}\nA:\n{options_text}")
                else:
                    qna_texts.append(f"Q: {current_question}\nA: {' '.join(current_answer_lines)}")
            current_question = question_match.group(1).strip()
            current_answer_lines = []
            options = {}
            continue

        if line.lower().startswith("answer:"):
            line = line[len("answer:"):].strip()
            if line:
                current_answer_lines.append(line)
            continue

        option_match = re.match(r"Option\s+(\d+):", line, re.IGNORECASE)
        if option_match:
            option_key = f"Option {option_match.group(1)}"
            option_text = line.split(":", 1)[1].strip()
            options[option_key] = option_text
            continue

        if options:
            last_key = list(options.keys())[-1]
            options[last_key] += " " + line
        else:
            current_answer_lines.append(line)

    # Final save
    if current_question:
        if options:
            options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
            qna_texts.append(f"Q: {current_question}\nA:\n{options_text}")
        else:
            qna_texts.append(f"Q: {current_question}\nA: {' '.join(current_answer_lines)}")

    return qna_texts


async def process_qna_document(qna_file_path):
    """Parse and save QnA as flattened text."""
    print("📄 Parsing QnA document...")
    qna_texts = parse_qna_txt_file(qna_file_path)
    save_qna_texts(qna_texts)


async def process_screenshots_and_store_context():
    """Capture and parse screenshots, saving results."""
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

    save_app_context(parsed_results)


async def process_all_context_data(qna_file_path):
    """Run full pipeline for screenshots and QnA."""
    await process_screenshots_and_store_context()
    await process_qna_document(qna_file_path)


async def main():
    await process_all_context_data(QNA_DOC_PATH)


if __name__ == "__main__":
    asyncio.run(main())
