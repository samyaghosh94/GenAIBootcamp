import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()


def load_webpage(url: str) -> List:
    """
    Load a webpage and process it into a list of Document objects.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def text_splitting(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (List[Document]): The list of documents to split.

    Returns:
        List[Document]: A list of smaller document chunks.
    """
    # Split documents recursively and create smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    web_documents = text_splitter.split_documents(docs)
    return web_documents


def embeddings(split_docs: List[Document]) -> Chroma:
    """
    Create vector embeddings using OpenAI embeddings and Chroma for a set of documents.

    Args:
        split_docs (List[Document]): A list of smaller document chunks.

    Returns:
        Chroma: A Chroma vector store containing the embeddings.
    """
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("DIAL_LAB_KEY"),
        api_version=os.getenv("AZURE_API_VERSION")

    )
    return Chroma.from_documents(split_docs, embeddings)


def retriever(query: str, vector_db: Chroma) -> List[Document]:
    """
    Perform a similarity search query in the FAISS vector database.

    Args:
        query (str): User query string.
        vector_db (FAISS): The FAISS vector store containing document embeddings.

    Returns:
        List[Document]: A list of Document objects most similar to the query.
    """
    retriever = vector_db.as_retriever()
    return retriever.invoke(query)


def chat(query: str, context_docs: List[Document]) -> str:
    """
    Generate a response using ChatOpenAI based on the context from retrieved documents.

    Args:
        query (str): User query string.
        context_docs (List[Document]): Documents retrieved from similarity search.

    Returns:
        str: Generated answer from the language model.
    """
    # Combine the content of retrieved documents to form a context
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Construct the prompt with context and query
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}
    
    Question:
    {query}
    
    Answer:"""

    print(prompt)
    print("===============================================")
    llm = AzureChatOpenAI(model="gpt-4o-mini-2024-07-18",
                          azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                          api_key=os.getenv("DIAL_LAB_KEY"),
                          api_version=os.getenv("AZURE_API_VERSION")
                          )
    response = llm.invoke(prompt)
    return response.content


if __name__ == '__main__':
    # Input: Ask the user for the path of the PDF file.
    url = input("Enter the Website URL: ")

    # Load and process the Website.
    web_docs = load_webpage(url)

    # Split the loaded documents into smaller chunks.
    split_docs = text_splitting(docs=web_docs)

    # Create embeddings for the split documents.
    vector_db = embeddings(split_docs)

    # Input: Ask the user for a query to search for.
    query = input("Enter your query: ")

    # Retrieve results from the vector database.
    context = retriever(query, vector_db)

    # Response from LLM
    response = chat(query, context)

    print(response)
