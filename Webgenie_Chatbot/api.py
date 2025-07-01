import json
import os
import uuid
from typing import Optional
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.messages import HandoffMessage, TextMessage, ThoughtEvent
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from constants import RAG_PROMPT, FEEDBACK_PROMPT, ROUTER_PROMPT
from tools.rag_tool import rag_tool
from tools.save_feedback_tool import save_feedback_tool
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
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


# === Request body ===
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


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
    handoffs=["rag_agent", "feedback_agent", "user"],
    system_message=ROUTER_PROMPT
)

rag_agent = AssistantAgent(
    "rag_agent",
    model_client=gemini_client,
    tools=[rag_tool],
    handoffs=["router_agent", "user"],
    system_message=RAG_PROMPT
)

feedback_agent = AssistantAgent(
    "feedback_agent",
    model_client=gemini_client,
    tools=[save_feedback_tool],  # A custom tool to save to MongoDB
    handoffs=["router_agent", "user"],
    system_message=FEEDBACK_PROMPT
)

termination = HandoffTermination(target="user")
team = Swarm([router_agent, rag_agent, feedback_agent], termination_condition=termination)


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
async def chat_api(body: QueryRequest):
    session_id = body.session_id or str(uuid.uuid4())
    user_query = body.query

    session_data = load_session(session_id)
    # last_agent = session_data.get("last_agent", "router_agent")
    last_agent = "router_agent"  # Always start with the router agent for routing
    message_history = session_data.get("messages", [])

    handoff_task = HandoffMessage(
        source="user",
        target=last_agent,
        content=user_query
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

        return {
            "response": last_text or "No relevant response generated.",
            "session_id": session_id
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id}
        )

# === Health check ===
@app.get("/status")
def status():
    return {"status": "Swarm RAG API is running ✅"}
