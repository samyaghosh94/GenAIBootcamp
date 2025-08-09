🧠 GenAI Chatbot Flow Overview (Swarm-based)
***This document outlines the architecture and data flow of the intelligent GenAI chatbot that uses screen context, QnA data, and file attachments to assist users via an agent-based system powered by Autogen's Swarm framework, Gemini API, and FastAPI.***

🔁 High-Level Chat Flow
+------------------+
|   User Input     | ---> POST /chat
+------------------+
        |
        v
+----------------------------+
| FastAPI Endpoint (/chat)  |
| - Accepts query, session  |
| - Receives HTML + file    |
+----------------------------+
        |
        v
+----------------------------+
|  Session Manager           |
|  - Load chat history       |
|  - Create or resume session|
+----------------------------+
        |
        v
+----------------------------+
|  Router Agent              |
|  (Detects intent from user |
|   prompt + screen context) |
+----------------------------+
     |         |        |
     v         v        v
 [rag_agent] [troubleshoot_agent] [helper_agent]
     |         |        |
     +-------> Swarm handles routing
                based on prompt + metadata

<img width="584" height="324" alt="Prompt Flow Diagram" src="https://github.com/user-attachments/assets/819e7815-9941-4afa-8792-de63ed734429" />


🤖 Agent Roles and Routing
| Agent                | Purpose                                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `router_agent`       | Classifies user intent and routes input to the correct agent based on keywords, HTML context, and attachment metadata. |
| `rag_agent`          | Retrieves feature-related information from a knowledge base using `rag_tool`.                                          |
| `troubleshoot_agent` | Diagnoses reported errors using HTML context and predefined heuristics; logs issues with `save_feedback_tool`.         |
| `helper_agent`       | Extracts structured data from uploaded `.msg` files and fills in required fields using `msg_extraction_tool`.          |

📥 Input Types Supported
✅ Natural language question or command
✅ HTML source of current screen
✅ File attachment (.msg Outlook files for employee data extraction)
✅ Optional session ID for chat continuity

🧱 Data Sources and Tools
| Component              | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `rag_tool`             | Retrieves contextual answers from indexed documentation                            |
| `error_diagnosis_tool` | Diagnoses errors using user message + HTML                                         |
| `msg_extraction_tool`  | Extracts employee details from uploaded `.msg` file                                |
| `save_feedback_tool`   | Logs unresolved issues after retry failures                                        |
| FAISS Vectorstore      | Built from screenshot and QnA text data (offline); used by `rag_tool`              |
| HTML context           | Provides screen-specific hints for routing and answers (not a primary data source) |

📡 API Endpoint
POST /chat

Request Form Data:
- query: user input
- html_context: screen HTML
- attachment: optional file
- session_id: optional session key

Response:
{
  "response": "Resolved response from the relevant agent",
  "session_id": "uuid-session-id"
}

🔍 Internal Processing Flow (FastAPI + Swarm)
User Input → /chat
    ↓
Load/Resume Session
    ↓
Create Combined Prompt:
    - User query
    - HTML context
    - File path (if uploaded)
    ↓
Send to `router_agent` via Swarm
    ↓
Swarm runs:
    → Handoff to appropriate agent
        → Uses tools (e.g., rag_tool, msg_extraction_tool)
        → Generates response
    ↓
Message history saved
    ↓
Return final message to user

<img width="351" height="349" alt="Component Diagram" src="https://github.com/user-attachments/assets/1c35277d-d237-4cd9-9e18-458ab83f22ae" />


💡 Detailed Agent Behaviors
- 🧭 router_agent:
Detects intent (feature help, errors, data extraction)
Parses HTML for on-screen error indicators (404, warnings, etc.)
Uses metadata like attachment_path to guide routing

- 📚 rag_agent:
Uses retrieve_context to get help articles, navigation links, and answers
Contextualizes user location from HTML
Provides concise responses + deep links where applicable

- 🛠️ troubleshoot_agent:
Diagnoses functional issues from user message and UI errors
Detects failed retries using phrase matching (e.g., “still not working”)
Logs issue feedback if problem persists

- 📩 helper_agent:
Parses .msg attachments for structured data
Cross-checks required fields via rag_tool
Returns extracted data and prompts user for missing inputs if needed

🗂️ Key File Responsibilities
| File                           | Role                                     |
| ------------------------------ | ---------------------------------------- |
| `api.py`                       | FastAPI app and entrypoint               |
| `tools/rag_tool.py`            | Doc-based QnA retriever                  |
| `tools/error_rag_tool.py`      | Troubleshooting context retriever        |
| `tools/msg_extraction_tool.py` | Extracts data from uploaded `.msg` files |
| `tools/save_feedback_tool.py`  | Logs unresolved error feedback           |
| `config.py`                    | Constants (e.g., attachment directory)   |
| `chat_history/`                | Stores session-wise message history      |
| `constants.py`                 | System prompts for agents                |

✅ Design Highlights
- 🤖 Multi-agent autonomy: Each agent performs only its intended task
- 🔁 Session-aware: Tracks history across multiple user messages
- 🧠 Context-aware: Uses screen HTML for better user understanding
- 📂 Tool-augmented: Tools handle retrieval, extraction, diagnosis
- ⚡ Modular and scalable: Easy to extend with new tools or agents
- 🔒 Safe and deterministic: No hallucination, controlled agent behavior
