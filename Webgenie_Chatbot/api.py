import json
import os
import uuid
from typing import Optional
from datetime import datetime, timezone
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.messages import HandoffMessage, TextMessage, ThoughtEvent
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from constants import RAG_PROMPT, TROUBLESHOOT_PROMPT, ROUTER_PROMPT, HELPER_PROMPT
from tools.rag_tool import rag_tool
from tools.save_feedback_tool import save_feedback_tool
from tools.error_rag_tool import error_diagnosis_tool
from tools.msg_extraction_tool import msg_extraction_tool
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from config import ATTACHMENT_DIR
load_dotenv()

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Storage for chat histories ===
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


# === Safe content formatting ===
def safe_text(content):
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        return " ".join(str(c) for c in content).strip()
    elif content is None:
        return ""
    else:
        return str(content).strip()

# === File handling ===
def save_attachment(file: UploadFile) -> Optional[str]:
    if file and file.filename.lower().endswith(".msg"):
        os.makedirs(ATTACHMENT_DIR, exist_ok=True)
        file_path = os.path.join(ATTACHMENT_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file_path
    return None


# # === Request body ===
# class QueryRequest(BaseModel):
#     query: str = Form(...),
#     html_context: Optional[str] = Form(""),
#     session_id: Optional[str] = Form(None),
#     attachment: Optional[UploadFile] = File(None)

# === OpenAI / Gemini client ===
gemini_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GENAI_PLUS"),
    model_info=ModelInfo(
        family="gemini",
        vision=False,
        function_calling=True,
        json_output=False,
    ),
)

# === Agents ===
router_agent = AssistantAgent(
    "router_agent",
    model_client=gemini_client,
    handoffs=["rag_agent", "troubleshoot_agent", "helper_agent", "user"],
    system_message=ROUTER_PROMPT
)

rag_agent = AssistantAgent(
    "rag_agent",
    model_client=gemini_client,
    tools=[rag_tool],
    handoffs=["router_agent", "user"],
    system_message=RAG_PROMPT
)

troubleshoot_agent = AssistantAgent(
    "troubleshoot_agent",
    model_client=gemini_client,
    tools=[save_feedback_tool, error_diagnosis_tool],  # A custom tool to save to MongoDB
    handoffs=["router_agent", "user"],
    system_message=TROUBLESHOOT_PROMPT
)

helper_agent = AssistantAgent(
    "helper_agent",
    model_client=gemini_client,
    tools=[msg_extraction_tool, rag_tool],
    handoffs=["router_agent", "user"],
    system_message=HELPER_PROMPT
)

termination = HandoffTermination(target="user")
team = Swarm([router_agent, rag_agent, troubleshoot_agent, helper_agent], termination_condition=termination)


# === Session file I/O ===
def get_history_path(session_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")


def load_session(session_id: str):
    path = get_history_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"last_agent": "router_agent", "messages": []}


def save_session(session_id: str, last_agent: str, messages: list):
    path = get_history_path(session_id)
    with open(path, "w") as f:
        json.dump({"last_agent": last_agent, "messages": messages}, f, indent=2)


# === Chat endpoint ===
@app.post("/chat")
async def chat_api(query: str = Form(...),
    html_context: Optional[str] = Form(""),
    session_id: Optional[str] = Form(None),
    attachment: Optional[UploadFile] = File(None),
                   ):
    session_id = session_id or str(uuid.uuid4())
    session_data = load_session(session_id)
    # last_agent = session_data.get("last_agent", "router_agent")
    last_agent = "router_agent"  # Always start with the router agent for routing
    message_history = session_data.get("messages", [])

    # Save file and get path
    attachment_path = save_attachment(attachment)
    print("attachment is:", attachment)
    if attachment:
        combined_prompt = f"""
        User Prompt:
        {query}
    
        User's current screen HTML source:
        {html_context}
        
        NOTE: User uploaded a file at {attachment_path}.This may be used for filling forms.
""".strip()
    else:
        combined_prompt = f"""
                User Prompt:
                {query}

                User's current screen HTML source:
                {html_context}                
                """.strip()


    handoff_task = HandoffMessage(
        source="user",
        target=last_agent,
        content=combined_prompt,
        metadata={"attachment_path": attachment_path} if attachment_path else {}
    )

    last_text = None
    last_handoff_source = None

    try:
        async for msg in team.run_stream(task=handoff_task):
            if isinstance(msg, (ThoughtEvent, TextMessage, HandoffMessage)):
                content_str = safe_text(msg.content)

                # ✅ Add this line for debugging
                print(
                    f"[DEBUG] Received from {type(msg).__name__} | source: {getattr(msg, 'source', None)} | content: {msg.content}")

                if isinstance(msg, (ThoughtEvent, TextMessage)):
                    if content_str:  # update only if non-empty
                        last_text = content_str
                        f"[DEBUG] Last text updated: {last_text}"

                if isinstance(msg, HandoffMessage):
                    last_handoff_source = msg.source

                # Save message details
                message_history.append({
                    "type": type(msg).__name__,
                    "content": content_str,
                    "source": getattr(msg, "source", None),
                    "target": getattr(msg, "target", None),
                })

        # Save session state for next request
        if last_handoff_source:
            save_session(session_id, last_handoff_source, message_history)

            # Attempt to parse the response as JSON
        # try:
        #     json_response = json.loads(last_text)
        #     f"[DEBUG] Parsed JSON response: {json_response}"
        # except (json.JSONDecodeError, TypeError) as e:
        #     print(f"[WARN] Could not parse response as JSON: {e}")
        #     json_response = last_text  # fallback to string if it's not JSON

        return {
            "response": last_text or "No relevant response generated.",
            "session_id": session_id
        }

    except Exception as e:
        print(f"[ERROR] Exception during processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id}
        )

# === Health check ===
@app.get("/status")
def status():
    return {"status": "Swarm RAG API is running ✅"}

# i guess samya can judge this but something liek as aobve controllers
# @app.post("/session/start")
# async def start_session():
#     session_id = str(uuid.uuid4())
#     return JSONResponse(content={"session_id": session_id})