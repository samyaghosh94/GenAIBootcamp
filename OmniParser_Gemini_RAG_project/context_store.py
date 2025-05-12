# context_store.py

import json
import os
from typing import Dict, List

CONTEXT_FILE = "parsed_context.json"


def save_context(context_data: Dict[str, str]):
    """Save the parsed context to disk."""
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(context_data, f, indent=2)
    print(f"💾 Context saved to {CONTEXT_FILE}")


def load_context() -> str:
    """Load the full context (concatenated semantics) from disk."""
    if not os.path.exists(CONTEXT_FILE):
        print(f"⚠️ Context file {CONTEXT_FILE} not found.")
        return ""

    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        context_dict = json.load(f)

    # Join all semantic chunks into one big context string
    return "\n\n".join(context_dict.values())
