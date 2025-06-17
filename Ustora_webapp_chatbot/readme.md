🧠 GenAI Chatbot Flow Overview
This document outlines the full architecture and data flow of your chatbot system that uses screenshot and QnA data, builds a combined FAISS vectorstore, and serves intelligent responses via a FastAPI endpoint.

High Level Flow - 

+-------------------+
|    User Input     |----> API Endpoint (POST /chat)
+-------------------+     
          |
          v
+---------------------+       +-----------------------------+
|   Load Chat History  | <--- |    Load/Parse Documents     |
|    (Previous Chats)  |     |  (QnA + Screenshot Data)    |
+---------------------+       +-----------------------------+
          |                                    |
          v                                    v
+----------------------------+   +----------------------------+
|   Add User Message to       |   |   Create Vectorstore       |
|   Chat History              |   |   (Embed QnA + Screenshot  |
+----------------------------+   |   Documents using DialLab) |
          |                                    |
          v                                    v
+----------------------------+   +----------------------------+
| LangChain Graph             |   |  FAISS Index Search       |
| (Processing chat history,   |   | (Using QnA + Screenshot   |
|  invoking retriever)        |   |  documents to find context)|
+----------------------------+   +----------------------------+
          |
          v
+----------------------------+
|   Generate AI Response     |
|   (From combined context)  |
+----------------------------+
          |
          v
+----------------------------+
| Save Response to Chat History|
+----------------------------+
          |
          v
+----------------------------+
| Return AI Response to User  |
+----------------------------+


## 2. Detailed Flow

The following sections describe each step in detail, from data ingestion to response delivery.

---

📸 1. Context Data Ingestion
        A. Screenshot Pipeline
        +-------------------+        +---------------------+         +-------------------------+
        | Web App UI Pages  | -----> | Screenshot Capture  | ----->  | Screenshot Images (.png)|
        +-------------------+        +---------------------+         +-------------------------+
                                                                                |
                                                                                v
                                                                +-----------------------------+
                                                                | Omniparser (OCR & Parsing)  |
                                                                +-----------------------------+
                                                                                |
                                                                                v
                                                                +-----------------------------+
                                                                | Parsed Text (JSON format)   | --> `DOCSTORE_PATH`
                                                                +-----------------------------+

        B. QnA Data Ingestion
        +----------------+         +---------------------------+
        |  qna.txt file  | ----->  | Text Parser (Q, A, Options)|
        +----------------+         +---------------------------+
                                                |
                                                v
                                +---------------------------+
                                |                           |
                                | Textified QnA JSON        | --> `QNA_PARSED_TEXT_PATH`
                                | (QNA_PARSED_TEXT_PATH)    |
                                +---------------------------+

🧱 2. Vectorstore Construction
All QnA and Screenshot data are loaded and embedded with **DialLab Embeddings**, stored in a **FAISS vectorstore**.

+------------------------+      +----------------------------+
| Screenshot Documents   |      | QnA Text Documents         |
| from DOCSTORE_PATH     |      | from QNA_PARSED_TEXT_PATH |
+------------------------+      +----------------------------+
              \                          /
               \                        /
                \                      /
           +----------------------------------+
           | Combine & Embed with DialLab     |
           | using FAISS + InMemoryDocstore   |
           +----------------------------------+
                        |
                        v
          +-----------------------------+
          | FAISS Index (vectorstore)   | --> Saved to disk - `FAISS_INDEX_PATH`
          +-----------------------------+

3. LangGraph Flow
+--------------------------------------+
| LangChain Graph Setup               |
| (Handles conversation flow)         |
|                                      |
| +-----------------------------+      |
| | Build Graph (session, state)|      |
| +-----------------------------+      |
|             |                        |
|             v                        |
| +-----------------------------+      |
| | LangChain Graph Execution   |      |
| | (Retrieve Context from FAISS)|      |
| | & Generate Response         |      |
| +-----------------------------+      |
|             |                        |
|             v                        |
| +-----------------------------+      |
| | Save AI Response to History |      |
| +-----------------------------+      |
+--------------------------------------+


🧠 4. Inference via FastAPI
+-------------+     /chat endpoint     +-------------------------+
| User Query  | ---------------------> | FastAPI App             |
+-------------+                        |                         |
                                       | 1. Load FAISS Vectorstore
                                       | 2. Retrieve context
                                       | 3. Call Gemini API (LangGraph)
                                       | 4. Return formatted reply
                                       +-------------------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |          AI Response        |
                                  +-----------------------------+

🧠 GenAI Chatbot – Detailed Architecture Overview
This chatbot intelligently answers questions by combining parsed screenshot data and QnA documents, embedding them into a FAISS vectorstore, and using Gemini API for reasoning. It is orchestrated using LangGraph and served through a FastAPI backend.

🔹 1. Context Data Preparation
a. Screenshot Pipeline
The system captures UI screenshots of your web application (automated via Selenium or similar).

These screenshots are then passed to OmniParser, which performs OCR and UI element parsing, extracting textual information from each screenshot.

The parsed text from all screenshots is saved in DOCSTORE_PATH as a list of JSON objects (each containing a "page_content" field).

b. QnA Text Document Pipeline
You maintain a qna.txt file with structured Q&A pairs (like Q1: ... A: ...).

This file is parsed to extract questions and answers (including options if applicable).

Two outputs are generated:

A structured dictionary (qna_data.json at QNA_PARSED_PATH) – useful for logic, debugging, or future extensions.

A flat list of text chunks like "Q: ...\nA: ..." saved to qna_texts.json at QNA_PARSED_TEXT_PATH, ready for embedding.

🔹 2. Embedding & Vectorstore Creation
Screenshot and QnA text documents are both loaded.

Using custom DialLab embeddings, each document is converted into a vector.

All vectors are stored in a single FAISS index, backed by an InMemoryDocstore, enabling semantic retrieval.

The index is saved to disk (FAISS_INDEX_PATH) for quick reloading in future sessions.

Why combined?
To let the chatbot answer queries by pulling from both app UI context and documentation seamlessly.

🔹 3. FastAPI Backend for Chat
A /chat/ endpoint receives user queries along with a session_id.

Chat history is loaded from disk (chat_history/ directory).

The vectorstore is loaded (or rebuilt if needed).

A retriever is initialized to fetch relevant context from the FAISS store.

This context, along with chat history and the current query, is sent to a LangGraph pipeline, which:

Constructs a prompt.

Calls the Gemini API for a response.

Returns a structured answer with possible action recommendations or control IDs.

🔹 4. LangGraph Flow
LangGraph maintains stateful interaction, with clearly defined nodes and edges:

It handles loading context, calling Gemini, parsing results, and returning structured responses.

Responses may include:

Explanation

Suggested control (e.g., "icon 11")

Action instructions

🔹 5. Design Highlights
✅ Modular design: Each step is its own file/function.

✅ Hybrid understanding: Screenshots + QnA.

✅ Persistent vectorstore and chat history.

✅ Asynchronous API for performance.

✅ Gemini integration for rich reasoning.

🔹 Key File Responsibilities
File	                                                            Role
context_data_setup.py	                        | Captures screenshots, parses QnA, and stores them.
vectorstore/loader.py	                        | Builds or loads FAISS vectorstore using DialLab embeddings.
vectorstore/custom_diallab_embeddings.py	| Handles DialLab model embeddings.
api.py	                                        | FastAPI server that handles chat endpoint and LangGraph calls.
graph/chat_graph.py	                        | Defines LangGraph flow for multi-step reasoning.
chat_history.py	                                | Saves and loads session-based chat history.

✅ Summary
This system allows your chatbot to intelligently reference both visual and textual data about your app, making it highly effective for answering user questions or guiding actions.
