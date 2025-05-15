import json
import os
from typing import List

from langchain_core.messages import messages_from_dict, BaseMessage

# Function to load the chat history from a file
def load_chat_history(session_id: str):
    chat_history_path = f"./.chat_memory/{session_id}_history.json"

    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as file:
            data = json.load(file)
            messages = data.get("messages", [])

            # Adjusting the format to match langchain_core.messages expected format
            for message in messages:
                if "data" not in message:
                    message["data"] = {
                        "content": message.get("content", ""),
                        "metadata": message.get("response_metadata", {})
                    }

                # Remove the additional fields that are not required by messages_from_dict
                message.pop("response_metadata", None)
                message.pop("additional_kwargs", None)
                message.pop("tool_calls", None)
                message.pop("invalid_tool_calls", None)
                message.pop("usage_metadata", None)

            return messages_from_dict(messages)
    else:
        return []

# Function to save chat history to a file
def save_chat_history(session_id: str, messages: List[BaseMessage]):
    os.makedirs("./.chat_memory", exist_ok=True)
    chat_history_path = f"./.chat_memory/{session_id}_history.json"

    # Convert the messages to a dictionary format that can be saved
    messages_dict = [msg.dict() for msg in messages]  # langchain message to dict

    with open(chat_history_path, "w", encoding="utf-8") as file:
        json.dump({"messages": messages_dict}, file, ensure_ascii=False, indent=4)
