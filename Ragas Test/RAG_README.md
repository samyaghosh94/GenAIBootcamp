# RAG Pipeline with Gemini

A complete Retrieval-Augmented Generation (RAG) pipeline using Google's Gemini model and FAISS vector store.

## Features

✨ **Core Capabilities:**
- Uses Google Gemini 1.5 Flash LLM for fast inference
- **FAISS vector store** for efficient similarity search
- **Gemini embeddings** for semantic search
- Modern LangChain LCEL (LangChain Expression Language) implementation
- Configurable temperature and retrieval parameters

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in this directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
GENAI_PLUS=your_google_api_key_here  # Same key for embeddings
```

**Getting Your Google API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API Key"
3. Create a new API key
4. Copy and paste it into your `.env` file

### 3. Vector Store Setup

Your FAISS index should already be created in the `./Data` folder from `generate_ragas_dataset.py`. The pipeline will automatically load it.

## Usage

### Quick Start

```python
from rag_pipeline import rag_pipeline

# Query the RAG pipeline
query = "What is artificial intelligence?"
response, docs = rag_pipeline(query, temperature=0.0, top_k=3)

print(f"Response: {response}")
print(f"Retrieved {len(docs)} documents")
```

### With Custom Parameters

```python
# Control temperature (0.0 = deterministic, 1.0 = creative)
response, docs = rag_pipeline(
    query="Your question here",
    temperature=0.5,
    top_k=5  # Retrieve top 5 documents
)
```

## Architecture

```
┌─────────────────────┐
│   User Query        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Retriever (FAISS + Gemini)    │  ◄─ Retrieves top-k relevant docs
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Prompt Template   │  ◄─ Formats context + question
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Gemini LLM 1.5     │  ◄─ Generates response
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Response          │
└─────────────────────┘
```

## Configuration

### Change Model

In `rag_pipeline.py`, modify the LLM initialization:

```python
# Use Gemini 1.5 Pro for better quality
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # or "gemini-1.5-flash"
    temperature=0.0,
    google_api_key=os.getenv("GENAI_PLUS")
)
```

### Change Retrieval Parameters

```python
# Modify retriever settings
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 docs
```

## Files

- **rag_pipeline.py** - Core RAG pipeline implementation
- **example_usage.py** - Usage examples with vector store setup
- **test_rag_pipeline.py** - Integration tests
- **requirements.txt** - Python dependencies
- **Data/faiss_index.index** - Your FAISS vector index
- **Data/documents.json** - Document store
- **RAG_README.md** - This documentation

## Troubleshooting

### "No module named 'langchain'" 
```bash
pip install langchain langchain-core langchain-community
```

### "GOOGLE_API_KEY not found"
- Ensure `.env` file is in the same directory as the script
- Check the API key is valid

### "FAISS index or docstore not found"
- Ensure `Data/faiss_index.index` and `Data/documents.json` exist
- Run `generate_ragas_dataset.py` first if missing

### "GeminiEmbeddings not found"
- Ensure the path to `gemini_embeddings.py` is correct
- Check that the Webgenie_Chatbot project exists

## Performance Tips

1. **For faster responses**: Use `gemini-1.5-flash` (default)
2. **For better quality**: Use `gemini-1.5-pro`
3. **Adjust retrieval**: Modify `top_k` parameter (3-5 is optimal)
4. **Temperature**: Use 0.0 for factual answers, 0.7-1.0 for creative

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Gemini Embeddings](https://ai.google.dev/docs/embeddings)
