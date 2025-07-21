import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
# from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()

# 2. Load or create your vector store retriever (example: FAISS)
# Suppose you have some docs loaded
loader = TextLoader(r"C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\Genai_ReAct\Harry_Potter_all_books_preprocessed.txt")  # your documents file
docs = loader.load()

# Create embeddings and vectorstore (FAISS example)
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small-1",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("DIAL_LAB_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
        )

# Assuming docs is a list of Document objects loaded by TextLoader
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Adjust size based on your use case
    chunk_overlap=50  # Optional: overlap between chunks
)

# Split your docs into smaller chunks
texts = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# 3. Initialize the language model (AzureOpenAI or OpenAI)
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2024-07-18",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("DIAL_LAB_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

# 4. Initialize LangGraph prebuilt QA agent with retriever and LLM
agent = create_react_agent(
    llm=llm,
    retriever=retriever,
    verbose=True,
)

# 5. Run the agent for a query
question = "Who is the main character in the Harry Potter series and what is his role in the story?"
response = agent.invoke({"input": question}, stream=True)

print("Answer:", response)