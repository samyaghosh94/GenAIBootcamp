import extract_msg
import spacy
import re
from typing import Dict, Any
import dateparser
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage
import json

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def msg_extraction_tool(file_path: str) -> dict[str, str | None | Any]:
    """
    Extract structured info from a .msg Outlook email file.
    Returns: dictionary with possible keys: name, email, phone, start_date, role, department
    """
    msg = extract_msg.Message(file_path)
    msg_sender = msg.sender or ""
    msg_to = msg.to or ""
    msg_subject = msg.subject or ""
    msg_body = msg.body or ""

    full_text = f"{msg_subject}\n{msg_body}"

    doc = nlp(full_text)

    result = {
        "from": msg_sender,
        "to": msg_to,
        "subject": msg_subject.strip(),
        "body": msg_body.strip(),
    }

    # === Name Extraction ===
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if names:
        result["name"] = names[0]

    # === Email Extraction ===
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(email_regex, full_text)
    if emails:
        result["email"] = emails[0]

    # === Phone Extraction ===
    phone_regex = r"\+?\d[\d\-\s]{7,}\d"
    phones = re.findall(phone_regex, full_text)
    if phones:
        result["phone"] = phones[0]

    # === Start Date Extraction ===
    for ent in doc.ents:
        if ent.label_ == "DATE":
            parsed_date = dateparser.parse(ent.text)
            if parsed_date:
                result["start_date"] = parsed_date.strftime("%Y-%m-%d")
                break

    # === Role / Department Guess ===
    department_keywords = ["HR", "Finance", "Engineering", "Sales", "Marketing", "Admin"]
    for dept in department_keywords:
        if dept.lower() in full_text.lower():
            result["department"] = dept
            break

    role_match = re.search(r"\b(Role|Title|Position):?\s*(\w+)", full_text, re.IGNORECASE)
    if role_match:
        result["role"] = role_match.group(2)

    # ✅ Return message with override_llm
    return result

msg_extraction_tool = FunctionTool(
    name="msg_extraction_tool",
    func=msg_extraction_tool,
    description="Extracts relevant information from a .msg Outlook email file.",
)

