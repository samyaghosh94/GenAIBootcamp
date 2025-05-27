import os
import json
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import trim_messages, messages_from_dict, messages_from_dict
from typing import TypedDict, List
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables from .env file
load_dotenv()

# Ensure the .chat_memory folder exists (create if it doesn't)
if not os.path.exists("./.chat_memory"):
    os.makedirs("./.chat_memory")

# Initialize Azure OpenAI model with environment variables
chat_model = init_chat_model(
    model="gpt-4o-mini-2024-07-18",
    model_provider="azure_openai",
    api_key=os.getenv("DIAL_LAB_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

# Define a message trimmer to limit the number of tokens
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=chat_model,
    include_system=True,
    allow_partial=True,
)


# Define the state of the chat (with message history)
class MessagesState(TypedDict):
    messages: List
    session_id: str  # Include session_id to link with memory saver


# Function to generate a response from the model
def generate_response(state: MessagesState) -> MessagesState:
    try:
        # Check the messages before trimming and passing to the model
        print(f"Messages before trimming: {state['messages']}")
        trimmed = trimmer.invoke(state["messages"])
        response = chat_model.invoke(trimmed)
        print("Model Response:", response)
        state["messages"].append(response)

        # Save the chat history to a file for the session
        save_chat_history(state["messages"], state["session_id"])

        return state
    except Exception as e:
        print(f"Error generating response: {e}")
        return state


# Function to save the chat history to a file
def save_chat_history(messages: List, session_id: str):
    from langchain_core.messages import messages_to_dict
    chat_history_path = f"./.chat_memory/{session_id}_history.json"

    # ✅ Convert all messages to dicts using LangChain utility
    serializable_messages = messages_to_dict(messages)

    with open(chat_history_path, "w") as file:
        json.dump({"session_id": session_id, "messages": serializable_messages}, file, indent=4)


# Function to load the chat history from a file
def load_chat_history(session_id: str) -> List:
    from langchain_core.messages import messages_from_dict
    chat_history_path = f"./.chat_memory/{session_id}_history.json"

    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as file:
            data = json.load(file)
            return messages_from_dict(data.get("messages", []))
    else:
        return []


# Create and configure the LangGraph workflow
def build_graph(session_id: str):
    # Load chat history if it exists
    initial_history = load_chat_history(session_id)

    # Define a new graph with the MessagesState schema
    workflow = StateGraph(state_schema=MessagesState)

    # Initialize state with the loaded chat history
    initial_state = {"messages": initial_history, "session_id": session_id}

    # Don't add the system message if the history is empty (avoid default message on start)
    if not initial_history:
        initial_state["messages"] = []  # Start with an empty list if no history exists

    # Define the function that calls the model
    def call_model(state: MessagesState):
        return generate_response(state)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory using MemorySaver and pass the session_id as checkpoint_id
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)  # Compile the graph with memory

    return app, initial_state
