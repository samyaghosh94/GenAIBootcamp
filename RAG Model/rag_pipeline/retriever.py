from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings

def build_faiss_index(texts, embedding_model):
    # If embedding_model is not wrapped in LangChain, wrap it with SentenceTransformerEmbeddings
    if not isinstance(embedding_model, SentenceTransformerEmbeddings):
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Document objects for each text
    docs = [Document(page_content=text) for text in texts]

    # Create FAISS index from the list of documents
    faiss_index = FAISS.from_documents(docs, embedding_model)

    return faiss_index
