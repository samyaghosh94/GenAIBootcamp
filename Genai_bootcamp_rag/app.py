import streamlit as st
from chatbot_graph import build_graph
from langchain.schema import HumanMessage, AIMessage
import uuid
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables
load_dotenv()

# Your modules
from data.web_data_loader import load_webpage, text_splitting
from embedder_model import embeddings
from retriever import retriever

# Streamlit page setup
st.set_page_config(page_title="RAG WebLoader Chatbot")
st.title("🧠 RAG WebLoader Chatbot")

# Session ID setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# Build LangGraph for this session
graph, _ = build_graph(session_id)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Traced function with clean prompt separation
@traceable(name="Process Web Chat Query")
def process_query_with_context(user_input, vector_db, graph, session_id, chat_history):
    # Step 1: Retrieve documents
    web_context = retriever(user_input, vector_db)

    # Step 2: Build context string
    context = "\n\n".join([doc.page_content for doc in web_context])

    # Step 3: Construct the internal prompt with context (not shown to user)
    internal_prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{user_input}

Answer:"""

    # Step 4: Add **only the raw user input** to chat history (for UI display)
    chat_history.append(HumanMessage(content=user_input))

    # Step 5: Create a model-facing history with the internal prompt
    model_chat_history = chat_history[:-1] + [HumanMessage(content=internal_prompt)]

    # Step 6: Invoke the LangGraph model
    result = graph.invoke(
        {
            "messages": model_chat_history,
            "session_id": session_id
        },
        config={"configurable": {"thread_id": session_id}}
    )

    return result

# UI: URL input
url = st.text_input("Enter the Website URL:")

if url:
    # Step 1: Load and split the webpage
    web_docs = load_webpage(url)
    split_docs = text_splitting(docs=web_docs)
    vector_db = embeddings(split_docs)

    # Step 2: User question
    user_input = st.chat_input("Ask a question about the website")

    if user_input:
        try:
            # Step 3: Process the query using the traced function
            result = process_query_with_context(
                user_input=user_input,
                vector_db=vector_db,
                graph=graph,
                session_id=session_id,
                chat_history=st.session_state.chat_history
            )

            # Step 4: Store model response (if valid)
            if "messages" in result and result["messages"]:
                st.session_state.chat_history = result["messages"]
            else:
                st.error("Error: Model returned an empty response.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display user and assistant messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
