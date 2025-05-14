import streamlit as st
from chatbot_graph import build_graph
from langchain.schema import HumanMessage, AIMessage
import uuid  # To generate unique session IDs

# Set up Streamlit page
st.set_page_config(page_title="Persistent Chatbot")
st.title("🧠 Persistent Chatbot with LangGraph")

# Generate a unique session ID if it doesn't exist (for new users)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Automatically generate a unique session ID

# Get the session ID for the current user
session_id = st.session_state.session_id

# Build the LangGraph instance for this session (using the auto-generated session_id)
graph, _ = build_graph(session_id)  # `_` is for unused initial_state

# Initialize chat history in session state (starting with an empty list or last conversation)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input field
user_input = st.chat_input("Say something")

if user_input:
    # Append user message to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Pass chat history to the LangGraph instance to generate a response
    result = graph.invoke(
        {
            "messages": st.session_state.chat_history,
            "session_id": st.session_state.session_id
        },
        config={"configurable": {"thread_id": st.session_state.session_id}}
    )

    # Update chat history with the generated response
    st.session_state.chat_history = result["messages"]

# Display chat messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)
