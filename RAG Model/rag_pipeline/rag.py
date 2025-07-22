from .crawler import get_internal_links, load_docs_from_links
from .embedder import get_embedding_model
from .retriever import build_faiss_index
from .generator import ask_gemini
from langchain.text_splitter import CharacterTextSplitter


def gemini_rag_pipeline(base_url, question, max_links=10):
    # Fetch internal links from base_url
    links = get_internal_links(base_url, max_links)
    # Load documents from those links
    docs = load_docs_from_links(links)

    # Split the documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create list of chunked texts for embedding
    split_texts = [doc.page_content for doc in split_docs]

    # Inspect the first 500 characters of each chunked document
    print("\n=== CHUNKED DOCUMENTS LOADED ===")
    for i, text in enumerate(split_texts):
        print(f"\n--- Document {i + 1} ---\n{text[:500]}...\n")

    if not split_texts:
        return "No content found to answer your question."

    # Get the embedding model
    embedding_model = get_embedding_model()

    # Build the FAISS index using the chunked texts
    index = build_faiss_index(split_texts, embedding_model)

    # Perform similarity search to find relevant documents
    similar_docs = index.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in similar_docs)

    # Generate an answer using the context
    return ask_gemini(question, context)
