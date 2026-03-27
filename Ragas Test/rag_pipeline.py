import os
import faiss
import json
from pathlib import Path

# # Set tiktoken cache directory before importing anything that uses it
# os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.getcwd(), ".tiktoken_cache")
# os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Pre-cache tiktoken encoding to avoid network issues during evaluation
try:
    import tiktoken
    print("🔄 Pre-caching tiktoken encoding...")
    encoding = tiktoken.get_encoding("cl100k_base")
    print("✅ Tiktoken encoding cached successfully")
except Exception as e:
    print(f"⚠️  Tiktoken pre-caching issue (may retry): {e}")

load_dotenv()

# Initialize OpenAI LLM
# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.0,
#     api_key=os.getenv("OPENAI_API_KEY")
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GENAI_PLUS"),
    convert_system_message_to_human=True
)

# # Initialize OpenAI embeddings
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# Initialize GeminiEmbeddings
import sys
sys.path.append(r"C:\dev\GenAIBootcamp\Webgenie_Chatbot\vectorstore")
from gemini_embeddings import GeminiEmbeddings
embeddings = GeminiEmbeddings(api_key=os.getenv("GENAI_PLUS"))

# Configuration paths (matching generate_ragas_dataset.py)
FAISS_INDEX_PATH = "./Data/faiss_index.index"
DOCSTORE_PATH = "./Data/documents.json"

def load_faiss_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load existing FAISS vector store from disk.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCSTORE_PATH):
        raise FileNotFoundError(f"FAISS index or docstore not found. Expected at {FAISS_INDEX_PATH} and {DOCSTORE_PATH}")
    
    print("🔄 Loading FAISS vector store from disk...")
    
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load docstore
    with open(DOCSTORE_PATH, 'r', encoding='utf-8') as f:
        docstore_dict = json.load(f)
    
    # Convert back to Document objects
    docstore = InMemoryDocstore()
    for key, doc_string in docstore_dict.items():
        # Parse the string representation of Document object
        # Format: "page_content='...' metadata={...}"
        try:
            # Extract page_content
            page_content_start = doc_string.find("page_content='") + len("page_content='")
            page_content_end = doc_string.find("'", page_content_start)
            page_content = doc_string[page_content_start:page_content_end]
            
            # Extract metadata
            metadata_start = doc_string.find("metadata=") + len("metadata=")
            metadata_str = doc_string[metadata_start:]
            metadata = eval(metadata_str)  # Safe since we control the data
            
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            docstore._dict[key] = doc
        except Exception as e:
            print(f"Warning: Could not parse document {key}: {e}")
            # Fallback: treat the whole string as page_content
            doc = Document(
                page_content=doc_string,
                metadata={'chunk_id': int(key), 'source_file': 'unknown'}
            )
            docstore._dict[key] = doc
    
    # Create index_to_docstore_id mapping
    index_to_docstore_id = {i: str(i) for i in range(len(docstore_dict))}
    
    # Recreate FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    print(f"✅ FAISS vector store loaded with {len(docstore_dict)} documents")
    return vectorstore

# Initialize vectorstore by loading existing FAISS index
vectorstore = load_faiss_vectorstore(embeddings)

# Create RAG prompt template
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)

# Create RAG chain using LCEL (LangChain Expression Language)
def format_docs(docs):
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

def get_context(query):
    """Get relevant documents for a query"""
    return vectorstore.similarity_search(query, k=2)

def generate_response(query, context):
    """Generate response using LLM"""
    prompt = rag_prompt.format(context=context, question=query)
    return llm.invoke(prompt).content

# Simplified RAG function
def rag_pipeline_simple(query, temperature=0.0, top_k=3):
    """
    Execute RAG pipeline to answer a query using retrieved documents.
    
    Args:
        query (str): The user question
        temperature (float): Temperature for LLM generation
        top_k (int): Number of documents to retrieve
    
    Returns:
        tuple: (response, docs) - Generated response and retrieved documents
    """
    # Get relevant documents
    docs = vectorstore.similarity_search(query, k=top_k)
    
    # Format context
    context = format_docs(docs)
    
    # Update LLM temperature
    llm.temperature = temperature
    
    # Generate response
    response = generate_response(query, context)
    
    return response, docs

# For backward compatibility
rag_pipeline = rag_pipeline_simple
