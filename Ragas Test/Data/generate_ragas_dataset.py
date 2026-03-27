import os
import json
import faiss
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# Import GeminiEmbeddings
import sys
sys.path.append(r"C:\dev\GenAIBootcamp\Webgenie_Chatbot\vectorstore")
from gemini_embeddings import GeminiEmbeddings

load_dotenv()

# Configuration
STORAGE_FOLDER = "./storage"
QUESTIONS_FILE = "./questions.json"
OUTPUT_DATASET_PATH = "./ragas_evaluation_dataset.json"
FAISS_INDEX_PATH = "./faiss_index.index"
DOCSTORE_PATH = "./documents.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_docx_files(storage_folder: str) -> List[Dict[str, Any]]:
    """
    Load all DOCX files from the storage folder.
    Returns a list of documents with their content and metadata.
    """
    documents = []
    storage_path = Path(storage_folder)

    if not storage_path.exists():
        print(f"❌ Storage folder not found: {storage_path}")
        return documents

    docx_files = list(storage_path.glob("*.docx"))
    print(f"📄 Found {len(docx_files)} DOCX files")

    for docx_file in docx_files:
        # Skip temporary files (start with ~$)
        if docx_file.name.startswith('~$'):
            continue
            
        try:
            print(f"📂 Loading: {docx_file.name}")
            loader = UnstructuredLoader(
                file_path=str(docx_file),
                strategy="hi_res",
                chunking_strategy="by_title",
                include_orig_elements=False
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = docx_file.name
                documents.append(doc)
            print(f"✅ Loaded {len(docs)} elements from {docx_file.name}")
        except Exception as e:
            print(f"❌ Error loading {docx_file.name}: {e}")

    return documents


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks for better context retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} chunks from documents")
    return chunks


def load_questions_from_file(questions_file: str) -> List[str]:
    """
    Load questions from a JSON file.
    Expected format: [{"question": "..."}, {"question": "..."}, ...]
    """
    if not Path(questions_file).exists():
        print(f"❌ Questions file not found: {questions_file}")
        return []
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list of dicts and list of strings
        questions = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'question' in item:
                    questions.append(item['question'])
                elif isinstance(item, str):
                    questions.append(item)
        
        print(f"✅ Loaded {len(questions)} questions from {questions_file}")
        return questions
    except Exception as e:
        print(f"❌ Error loading questions file: {e}")
        return []

def load_questions_with_ground_truth(questions_file: str) -> List[Dict[str, str]]:
    """
    Load questions and ground_truth from JSON file.
    Expected format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    if not Path(questions_file).exists():
        print(f"❌ Questions file not found: {questions_file}")
        return []
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'question' in item and 'ground_truth' in item:
                    items.append({
                        'question': item['question'],
                        'ground_truth': item['ground_truth']
                    })
        
        print(f"✅ Loaded {len(items)} question-ground_truth pairs from {questions_file}")
        return items
    except Exception as e:
        print(f"❌ Error loading questions file: {e}")
        return []

def create_faiss_vectorstore(chunks: List[Dict[str, Any]], embeddings: GeminiEmbeddings) -> FAISS:
    """
    Create FAISS vector store from document chunks using Gemini embeddings.
    """
    print("🔄 Creating FAISS vector store from chunks...")
    
    # Convert chunks to LangChain Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                'chunk_id': i,
                'source_file': chunk.metadata.get('source_file', 'unknown')
            }
        )
        documents.append(doc)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the index and docstore
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)
    
    # Save docstore
    docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
    with open(DOCSTORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(docstore_dict, f, indent=2, default=str)
    
    print(f"✅ FAISS vector store created with {len(documents)} documents")
    print(f"   Index saved to: {FAISS_INDEX_PATH}")
    print(f"   Docstore saved to: {DOCSTORE_PATH}")
    
    return vectorstore


def load_faiss_vectorstore(embeddings: GeminiEmbeddings) -> FAISS:
    """
    Load existing FAISS vector store from disk.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCSTORE_PATH):
        raise FileNotFoundError("FAISS index or docstore not found. Run create_faiss_vectorstore first.")
    
    print("🔄 Loading FAISS vector store from disk...")
    
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load docstore
    with open(DOCSTORE_PATH, 'r', encoding='utf-8') as f:
        docstore_dict = json.load(f)
    
    # Convert back to Document objects
    docstore = InMemoryDocstore()
    for key, doc_data in docstore_dict.items():
        doc = Document(
            page_content=doc_data['page_content'],
            metadata=doc_data['metadata']
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


def find_relevant_contexts(question: str, vectorstore: FAISS, top_k: int = 3) -> List[str]:
    """
    Find relevant contexts for a question using FAISS similarity search.
    """
    try:
        # Perform similarity search
        docs = vectorstore.similarity_search(question, k=top_k)
        
        # Extract page content from retrieved documents
        contexts = [doc.page_content for doc in docs]
        
        return contexts
    except Exception as e:
        print(f"❌ Error retrieving contexts for question: {e}")
        return []


def create_ragas_dataset(vectorstore: FAISS, question_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Create a RAGAS evaluation dataset using FAISS retrieval.

    RAGAS dataset format:
    {
        "question": str (from JSON),
        "ground_truth": str (from JSON - expected answer),
        "contexts": List[str] (retrieved using FAISS similarity search)
    }
    """
    dataset = []

    if not vectorstore:
        print("❌ No vectorstore provided to create dataset")
        return dataset

    if not question_pairs:
        print("❌ No question-ground_truth pairs provided")
        return dataset

    print(f"\n📊 Processing questions with FAISS retrieval")

    # Create evaluation samples for each question
    for item in question_pairs:
        question = item['question']
        ground_truth = item['ground_truth']

        # Find relevant contexts for this question using FAISS
        contexts = find_relevant_contexts(question, vectorstore, top_k=2)

        # Only add if we found relevant contexts
        if contexts:
            dataset_item = {
                "question": question,
                "ground_truth": ground_truth,  # ✅ From JSON, not auto-generated!
                "contexts": contexts
            }
            dataset.append(dataset_item)
            print(f"  ✅ Added: {question[:60]}...")
        else:
            print(f"  ⚠️ Skipped (no relevant contexts): {question[:60]}...")

    return dataset


def save_ragas_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """
    Save the RAGAS dataset to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ RAGAS dataset saved to: {output_path}")
    print(f"📊 Dataset contains {len(dataset)} evaluation samples")


def main():
    """
    Main function to generate RAGAS evaluation dataset from DOCX files and questions.
    """
    print("🚀 Starting RAGAS Dataset Generation with FAISS Retrieval")
    print("=" * 60)
    
    # Step 1: Load questions with ground_truth from JSON file
    print("\n[Step 1] Loading questions with ground_truth from JSON...")
    question_pairs = load_questions_with_ground_truth(QUESTIONS_FILE)
    
    if not question_pairs:
        print("❌ No questions loaded. Exiting.")
        return
    
    # Step 2: Load DOCX files
    print("\n[Step 2] Loading DOCX files...")
    documents = load_docx_files(STORAGE_FOLDER)
    
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    print(f"✅ Total documents loaded: {len(documents)}")
    
    # Step 3: Chunk documents
    print("\n[Step 3] Chunking documents...")
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if not chunks:
        print("❌ No chunks created. Exiting.")
        return
    
    # Step 4: Create FAISS vector store with Gemini embeddings
    print("\n[Step 4] Creating FAISS vector store with Gemini embeddings...")
    embeddings = GeminiEmbeddings(api_key=os.getenv("GENAI_PLUS"))
    vectorstore = create_faiss_vectorstore(chunks, embeddings)
    
    # Step 5: Create RAGAS dataset using FAISS retrieval
    print("\n[Step 5] Creating RAGAS evaluation dataset with FAISS retrieval...")
    dataset = create_ragas_dataset(vectorstore, question_pairs)
    
    if not dataset:
        print("❌ No dataset items created. Exiting.")
        return
    
    # Step 6: Save dataset
    print("\n[Step 6] Saving dataset...")
    save_ragas_dataset(dataset, OUTPUT_DATASET_PATH)
    
    # Print sample
    print("\n📋 Sample dataset item:")
    if dataset:
        print(json.dumps(dataset[0], indent=2))
    
    print("\n" + "=" * 60)
    print("✅ RAGAS dataset generation with FAISS retrieval completed successfully!")
    print(f"📊 Dataset saved to: {OUTPUT_DATASET_PATH}")
    print(f"🔍 FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"📚 Docstore saved to: {DOCSTORE_PATH}")


if __name__ == "__main__":
    main()

