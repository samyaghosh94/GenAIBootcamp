# RAGAS Dataset Generation Guide

## Updated Workflow

Your RAGAS dataset generation has been updated to use **user-provided questions** instead of auto-generating them. This gives you full control over the evaluation samples.

## How It Works

### 1. **Prepare Your Questions**

Create a `questions.json` file in the Data folder with your questions:

```json
[
    {
        "question": "How do I add a new employee to the system?"
    },
    {
        "question": "What are the steps to process employee onboarding?"
    },
    {
        "question": "How can I update employee information in the portal?"
    }
]
```

**Note:** The function supports both formats:
- Format 1: `[{"question": "..."}, {"question": "..."}]` (Recommended)
- Format 2: `["question 1", "question 2"]`

### 2. **Run the Dataset Generator**

```bash
cd "C:\dev\GenAIBootcamp\Ragas Test\Data"
python generate_ragas_dataset.py
```

Or using the dataset.py file:

```bash
python dataset.py
```

### 3. **Output**

The script will generate `ragas_evaluation_dataset.json` with the following structure:

```json
[
    {
        "question": "How do I add a new employee to the system?",
        "ground_truth": "To add a new employee, first navigate to the employee management section...",
        "contexts": [
            "First most relevant chunk from document...",
            "Second relevant chunk...",
            "Third relevant chunk..."
        ]
    },
    ...
]
```

## File Configuration

The default configuration in both scripts:

```python
STORAGE_FOLDER = "./storage"              # Your DOCX files location
QUESTIONS_FILE = "./questions.json"       # Your questions file
OUTPUT_DATASET_PATH = "./ragas_evaluation_dataset.json"  # Output location
CHUNK_SIZE = 500                          # Characters per chunk
CHUNK_OVERLAP = 100                       # Overlap between chunks
```

## Processing Steps

1. **Load Questions** - Reads from `questions.json`
2. **Load Documents** - Loads DOCX file(s) from storage folder
3. **Chunk Documents** - Splits documents into 500-character chunks with 100-char overlap
4. **Match Questions to Contexts** - For each question, finds the 3 most relevant chunks
5. **Create Dataset Items** - Generates RAGAS format with ground truth
6. **Save Dataset** - Writes to `ragas_evaluation_dataset.json`

## Context Relevance Matching

The system finds the most relevant chunks for each question by:

1. Converting question to lowercase words set
2. Comparing with all chunk words
3. Calculating word overlap score
4. Taking the top 3 chunks with highest overlap

This ensures each question has the best matching context from your documents.

## Using the Generated Dataset

### With RAGAS Framework

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas_dataset_utils import RAGASDatasetUtils

# Load your generated dataset
dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")

# Evaluate your RAG system
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
)

print(results)
```

### Splitting into Train/Test

```python
from ragas_dataset_utils import RAGASDatasetUtils

# Load dataset
dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")

# Split 80/20
train_set, eval_set = RAGASDatasetUtils.split_dataset(dataset, train_ratio=0.8)

# Save separately
RAGASDatasetUtils.save_dataset(train_set, "train_dataset.json")
RAGASDatasetUtils.save_dataset(eval_set, "eval_dataset.json")
```

### Get Dataset Statistics

```python
from ragas_dataset_utils import RAGASDatasetUtils

dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")
stats = RAGASDatasetUtils.get_dataset_stats(dataset)

print(f"Total items: {stats['total_items']}")
print(f"Total contexts: {stats['total_contexts']}")
print(f"Avg contexts per item: {stats['avg_contexts_per_item']}")
```

## Example: Complete Workflow

### Step 1: Create questions.json
```json
[
    {"question": "How do I log in to the portal?"},
    {"question": "What should I do if I forget my password?"},
    {"question": "How can I view my payslip?"},
    {"question": "What is the process for requesting time off?"},
    {"question": "How do I update my profile information?"}
]
```

### Step 2: Run the generator
```bash
python generate_ragas_dataset.py
```

### Expected Output
```
🚀 Starting RAGAS Dataset Generation
==================================================

[Step 1] Loading questions from JSON file...
✅ Loaded 5 questions from ./questions.json

[Step 2] Loading DOCX files...
📄 Found [PosixPath('storage/HCM_Portal_User_Guide.docx')] DOCX file
📂 Loading: HCM_Portal_User_Guide.docx
✅ Loaded 45 elements from HCM_Portal_User_Guide.docx
✅ Total documents loaded: 45

[Step 3] Chunking documents...
✂️ Created 234 chunks from documents

[Step 4] Creating RAGAS evaluation dataset...
📊 Processing 234 chunks with 5 questions
  ✅ Added question: How do I log in to the portal?
  ✅ Added question: What should I do if I forget my password?
  ...

[Step 5] Saving dataset...
✅ RAGAS dataset saved to: ./ragas_evaluation_dataset.json
📊 Dataset contains 5 evaluation samples

==================================================
✅ RAGAS dataset generation completed successfully!
```

## Customization

### Modify Chunk Size

For finer-grained contexts, reduce chunk size:

```python
# In generate_ragas_dataset.py
CHUNK_SIZE = 300        # Smaller chunks
CHUNK_OVERLAP = 50      # Less overlap
```

### Modify Chunk Overlap

For more overlapping context:

```python
# In generate_ragas_dataset.py
CHUNK_SIZE = 1000       # Larger chunks
CHUNK_OVERLAP = 300     # More overlap (30%)
```

### Use Multiple Questions Files

```python
# Load multiple question files
questions1 = load_questions_from_file("questions_batch1.json")
questions2 = load_questions_from_file("questions_batch2.json")

# Merge
all_questions = questions1 + questions2

# Process
dataset = create_ragas_dataset(chunks, all_questions)
```

## Troubleshooting

### Issue: "Questions file not found"
- **Solution**: Make sure `questions.json` exists in the current directory
- Check file path with: `ls questions.json` (or `dir questions.json` on Windows)

### Issue: "No dataset items created"
- **Reason**: Questions don't match document content well
- **Solution**: 
  - Review your questions - make sure they're relevant to the documents
  - Check chunk_overlap setting
  - Increase CHUNK_SIZE to get larger contexts

### Issue: Dataset has fewer items than questions
- **Reason**: Some questions had no relevant chunks (overlap score = 0)
- **Solution**:
  - Make questions more specific to document content
  - Reduce CHUNK_SIZE to get more granular chunks
  - Check if documents are loading correctly

### Issue: Contexts are too similar
- **Reason**: CHUNK_OVERLAP is too high
- **Solution**: Reduce CHUNK_OVERLAP value
```python
CHUNK_OVERLAP = 50      # Reduce from 100
```

## Files Reference

- `generate_ragas_dataset.py` - Main script (standalone)
- `dataset.py` - Alternative script with same functionality
- `ragas_dataset_utils.py` - Utility functions
- `questions.json` - Your questions file (create this)
- `ragas_evaluation_dataset.json` - Generated output (automatic)
- `questions_example.json` - Example format reference

## Next Steps

1. ✅ Create your `questions.json` with your specific questions
2. ✅ Run `python generate_ragas_dataset.py`
3. ✅ Validate with: `RAGASDatasetUtils.validate_dataset()`
4. ✅ Use with RAGAS: `evaluate(dataset=your_dataset, ...)`

